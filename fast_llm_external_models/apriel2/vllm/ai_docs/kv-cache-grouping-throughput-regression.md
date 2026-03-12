# KV Cache Grouping and the Singleton-Layer Throughput Regression

How vLLM's KV cache grouping algorithm interacts with hybrid supernet models, why singleton mixer layers cause catastrophic throughput regression, and how we fixed it.

**Prerequisite**: [VLLM.md](VLLM.md) covers the basic KV cache concepts (pages, blocks, specs, groups, block tables). This document assumes you know what a `KVCacheSpec`, `KVCacheGroupSpec`, and block table are.

---

## Table of Contents

1. [The Symptom](#the-symptom)
2. [Background: The Grouping Algorithm](#background-the-grouping-algorithm)
3. [How group_size Is Computed](#how-group_size-is-computed)
4. [What group_size Controls](#what-group_size-controls)
5. [The Per-Forward Metadata Overhead](#the-per-forward-metadata-overhead)
6. [Root Cause: Spec Equality and group_size Degeneration](#root-cause-spec-equality-and-group_size-degeneration)
7. [Worked Examples](#worked-examples)
8. [Failed Approach: UnifiedMambaSpec](#failed-approach-unifiedmambaspec)
9. [Why UnifiedMambaSpec Failed](#why-unifiedmambaspec-failed)
10. [The Actual Fix: Monkey-Patching the Grouping Function](#the-actual-fix-monkey-patching-the-grouping-function)
11. [The Sliding Window Case](#the-sliding-window-case)
12. [Runtime Crash: Missing PIECEWISE Splitting Op](#runtime-crash-missing-piecewise-splitting-op)
13. [Runtime Crash: Triton constexpr JIT on Large Prefill Batches](#runtime-crash-triton-constexpr-jit-on-large-prefill-batches)
14. [Runtime Crash: Import-by-Value Bug in cudagraph_capturing_enabled](#runtime-crash-import-by-value-bug-in-cudagraph_capturing_enabled)
15. [Key Takeaways and Learnings](#key-takeaways-and-learnings)

---

## The Symptom

Throughput measurements for a 24-layer Apriel2 0.5B supernet with different placements:

| Placement | Layers | tok/s | Relative |
|-----------|--------|------:|----------|
| `a8_g8_k8` | 8 attention + 8 GDN + 8 KDA | **30,500** | 1.00x |
| `a12_g6_k6` | 12 attention + 6 GDN + 6 KDA | **30,000** | 0.98x |
| `a12_k12` | 12 attention + 12 KDA | **27,400** | 0.90x |
| `a12_g1_k11` | 12 attention + 1 GDN + 11 KDA | **11,700** | 0.38x |

The `a12_g1_k11` layout is **2.6x slower** than `a12_g6_k6`, despite being nearly identical in compute (just swapped 5 KDA layers for 5 GDN layers). The only structural difference: one mixer type has a **singleton** layer count.

This pattern was consistent: any layout with 1-2 layers of any mixer type suffered >2x throughput regression.

---

## Background: The Grouping Algorithm

Recall from [VLLM.md](VLLM.md) that vLLM groups layers by `KVCacheSpec` type so that layers in the same group share a block table. The question is: how does vLLM decide which layers go into which group, and how many groups are there?

The grouping happens in `_get_kv_cache_groups_uniform_page_size()` in `vllm/v1/core/kv_cache_utils.py`. This function handles **hybrid models** where layers have different spec types but the same `page_size_bytes` (ensured by our `unify_block_page_sizes()`).

### Step 1: Group by spec equality

```python
same_type_layers: dict[KVCacheSpec, list[str]] = defaultdict(list)
for layer_name, layer_spec in kv_cache_spec.items():
    same_type_layers[layer_spec].append(layer_name)
```

This uses Python's `dict` key lookup, which relies on `__hash__` and `__eq__` of the `KVCacheSpec` objects. Layers with equal specs end up in the same bucket.

**What "equal" means for frozen dataclasses**: Python auto-generates `__eq__` that compares ALL fields. For `MambaSpec`, this includes `shapes`, `dtypes`, `block_size`, `page_size_padded`, `mamba_type`, `num_speculative_blocks`. Two specs are equal only if every single field matches.

### Step 2: Compute group_size

This is where the trouble starts.

---

## How group_size Is Computed

```python
min_num_layers = min(len(layers) for layers in same_type_layers.values())
group_size = min_num_layers

max_num_layers = max(len(layers) for layers in same_type_layers.values())
if max_num_layers < min_num_layers * 1.25:
    group_size = max_num_layers
```

In words:

1. Find the **smallest** type (fewest layers of any single spec type)
2. Set `group_size` to that minimum
3. Exception: if max and min are within 25% of each other, use the max (to avoid excessive padding for nearly-balanced distributions)

The FIXME comment in the code explains why:

> At the moment of writing this code (2025-06-02), all open-source hybrid models follow a n:1 pattern between different attention types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and full), so we can use the "1" in the n:1 pattern as the group size.

This was designed for models like Gemma3 (5 sliding-window : 1 full-attention). For such models, `group_size = 1` means "each pattern repeat has 1 full-attention layer and 5 sliding-window layers, bundled into 6 groups of 1 layer each." That's fine - the layers follow a strict repeating pattern.

But Apriel2 is a supernet where the mixer type per layer is **arbitrary** (chosen by placement optimization, not a fixed repeating pattern). The n:1 assumption doesn't hold.

---

## What group_size Controls

`group_size` determines three things:

### 1. Number of physical KV cache tensors

```python
for i in range(group_size):
    shared_by = [groups[j].layer_names[i] for j if i < len(groups[j])]
    kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))
```

With `group_size = N`, vLLM creates N physical tensors. The i-th layer from each type shares tensor `i`.

### 2. Number of blocks per tensor

```python
num_blocks = available_memory // page_size // group_size
```

Total memory is split evenly across `group_size` tensors, so each tensor gets `num_blocks` blocks.

### 3. Number of KV cache groups

Each spec type is split into `ceil(num_layers / group_size)` sub-groups. The total number of `KVCacheGroupSpec` objects is:

```
total_groups = sum(ceil(count / group_size) for count in type_layer_counts)
```

**This is the critical number** -- it determines how many block tables and metadata builders are needed.

---

## The Per-Forward Metadata Overhead

Every forward pass, vLLM must build attention metadata for **each** KV cache group:

```python
# Simplified from gpu_model_runner.py
for kv_cache_gid in range(len(kv_cache_groups)):
    # 1. Fetch block table for this group (~GPU tensor op)
    block_table, slot_mapping = _get_block_table_and_slot_mapping(kv_cache_gid)

    # 2. For each AttentionGroup within this KV cache group:
    for attn_gid in range(len(attn_groups[kv_cache_gid])):
        # 3. Build metadata (slot mappings, sequence info, padding for CUDA graphs)
        builder = attn_groups[kv_cache_gid][attn_gid].get_metadata_builder()
        metadata = builder.build(block_table, slot_mapping, ...)
```

Each metadata build involves:

- CPU work: iterating over active requests, computing slot mappings
- GPU work: creating/copying tensors for block tables, state indices, padding
- Estimated cost: **~0.5ms per group** (varies with batch size)

For a **decode step** on a 0.5B model, the actual matrix multiplications take only ~2-5ms. The metadata overhead is the bottleneck.

### The overhead equation

```
forward_time = compute_time + (num_groups * metadata_cost_per_group)
throughput = batch_size / forward_time
```

If `metadata_cost_per_group = 0.5ms`:

| Config | num_groups | metadata_ms | compute_ms | total_ms | relative |
|--------|-----------|------------|-----------|---------|----------|
| `a12_k12` | 2 | 1.0 | 4.0 | 5.0 | 1.00x |
| `a12_g6_k6` | 4 | 2.0 | 4.0 | 6.0 | 0.83x |
| `a12_g1_k11` | **24** | **12.0** | 4.0 | **16.0** | **0.31x** |

This matches the observed 2.6x regression.

---

## Root Cause: Spec Equality and group_size Degeneration

For Apriel2, the `get_kv_cache_spec()` methods returned:

| Mixer | Spec type | Key differences |
|-------|-----------|----------------|
| Attention | `FullAttentionSpec` | Different class from MambaSpec |
| Sliding Window | `SlidingWindowSpec` | Different class from FullAttentionSpec |
| GDN | `MambaSpec(shapes=2-tuple, ...)` | 2 state tensors (conv + recurrent) |
| KDA | `MambaSpec(shapes=4-tuple, ...)` | 4 state tensors (3 conv + recurrent) |

GDN and KDA both use `MambaSpec` with `mamba_type="gdn_attention"`, the same backend (`GDNAttentionBackend`), and the same unified `page_size_padded`. But because `MambaSpec` is a frozen dataclass, Python's auto-generated `__eq__` compares **all** fields -- including `shapes` and `dtypes`. Since `(a, b) != (a, b, c, d)`, GDN and KDA specs are never equal.

This means `same_type_layers` has **three** entries for a model with all three mixer types:

```python
same_type_layers = {
    FullAttentionSpec(...): ["attn_0", "attn_1", ..., "attn_11"],  # 12 layers
    MambaSpec(shapes=2-tuple): ["gdn_0"],                          # 1 layer
    MambaSpec(shapes=4-tuple): ["kda_0", ..., "kda_10"],           # 11 layers
}
```

Then:

- `min_num_layers = 1` (from GDN)
- `max_num_layers = 12` (from attention)
- `12 < 1 * 1.25`? **No.** So `group_size = 1`.

With `group_size = 1`:

- Attention: `ceil(12/1) = 12` sub-groups
- GDN: `ceil(1/1) = 1` sub-group
- KDA: `ceil(11/1) = 11` sub-groups
- **Total: 24 KV cache groups**

Every single layer becomes its own group, each with its own block table and metadata builder.

---

## Worked Examples

### `a12_k12` (12 attention + 12 KDA) -- FAST

```
same_type_layers:
  FullAttentionSpec: 12 layers
  MambaSpec(4-tuple): 12 layers

min = 12, max = 12
12 < 12 * 1.25 = 15?  YES --> group_size = 12

Groups:
  Group 0 (attention): [attn_0 ... attn_11]   -- 12 layers
  Group 1 (KDA):       [kda_0 ... kda_11]     -- 12 layers

Total: 2 groups
Tensors: 12 (each shared by 1 attention + 1 KDA)
num_blocks = available_memory / page_size / 12
```

2 metadata builds per forward. Fast.

### `a12_g6_k6` (12 attention + 6 GDN + 6 KDA) -- FAST

```
same_type_layers:
  FullAttentionSpec:   12 layers
  MambaSpec(2-tuple):  6 layers  (GDN)
  MambaSpec(4-tuple):  6 layers  (KDA)

min = 6, max = 12
12 < 6 * 1.25 = 7.5?  NO --> group_size = 6

Groups:
  Attention sub-group 0: [attn_0 ... attn_5]   -- 6 layers
  Attention sub-group 1: [attn_6 ... attn_11]  -- 6 layers
  GDN group:             [gdn_0 ... gdn_5]     -- 6 layers
  KDA group:             [kda_0 ... kda_5]      -- 6 layers

Total: 4 groups
Tensors: 6 (each shared by 2 attention + 1 GDN + 1 KDA)
num_blocks = available_memory / page_size / 6
```

4 metadata builds per forward. Still fast.

### `a12_g1_k11` (12 attention + 1 GDN + 11 KDA) -- SLOW

```
same_type_layers:
  FullAttentionSpec:   12 layers
  MambaSpec(2-tuple):  1 layer   (GDN)  <-- singleton!
  MambaSpec(4-tuple):  11 layers (KDA)

min = 1, max = 12
12 < 1 * 1.25 = 1.25?  NO --> group_size = 1

Groups:
  12 attention sub-groups (1 layer each)
  1 GDN sub-group (1 layer)
  11 KDA sub-groups (1 layer each)

Total: 24 groups  <-- CATASTROPHIC
Tensors: 1 (shared by all 24 layers)
num_blocks = available_memory / page_size / 1
```

24 metadata builds per forward. The overhead dominates decode on a 0.5B model.

### `a8_g8_k8` (8 attention + 8 GDN + 8 KDA) -- FAST

```
same_type_layers:
  FullAttentionSpec:   8 layers
  MambaSpec(2-tuple):  8 layers (GDN)
  MambaSpec(4-tuple):  8 layers (KDA)

min = 8, max = 8
8 < 8 * 1.25 = 10?  YES --> group_size = 8

Groups:
  Group 0 (attention): [attn_0 ... attn_7]
  Group 1 (GDN):       [gdn_0 ... gdn_7]
  Group 2 (KDA):       [kda_0 ... kda_7]

Total: 3 groups
```

3 metadata builds per forward. Fast.

---

## Failed Approach: UnifiedMambaSpec

Our first idea: since GDN and KDA differ only in `shapes` and `dtypes` but share `page_size_bytes`, `block_size`, and `mamba_type`, we could create a `UnifiedMambaSpec` subclass with custom `__eq__`/`__hash__` that ignores `shapes`/`dtypes`:

```python
@dataclass(frozen=True, eq=False)
class UnifiedMambaSpec(MambaSpec):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MambaSpec):
            return NotImplemented
        return (
            self.page_size_bytes == other.page_size_bytes
            and self.mamba_type == other.mamba_type
            and self.block_size == other.block_size
        )

    def __hash__(self) -> int:
        return hash((self.page_size_bytes, self.mamba_type, self.block_size))
```

This would make GDN and KDA appear as the same type for grouping, reducing 24 groups to 2.

---

## Why UnifiedMambaSpec Failed

**Error**: `ValueError: not enough values to unpack (expected 4, got 2)` at `modeling_apriel2.py:2311` during CUDA graph capture.

The assumption that only per-layer specs are used for cache initialization was **wrong**. There's a critical second code path: `_reshape_kv_cache_tensors` in `gpu_model_runner.py`.

### The hidden pitfall: group spec vs per-layer spec

`initialize_kv_cache_tensors()` has TWO phases:

**Phase 1 — Raw tensor allocation** (per-layer spec): Works correctly.

```python
for layer_name, raw_tensor in kv_cache_raw_tensors.items():
    kv_cache_spec = kv_cache_specs[layer_name]  # <-- per-layer spec ✓
```

**Phase 2 — `_reshape_kv_cache_tensors`** (GROUP spec): **THIS IS THE PROBLEM.**

```python
for group in kv_cache_groups:
    kv_cache_spec = group.kv_cache_spec  # <-- GROUP spec, NOT per-layer!
    if isinstance(kv_cache_spec, MambaSpec):
        for layer_name in group.layer_names:
            state_tensors = []
            for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                # Uses GROUP spec shapes for ALL layers in group!
                tensor = torch.as_strided(raw_tensor.view(dtype), ...)
                state_tensors.append(tensor)
            kv_caches[layer_name] = state_tensors
```

When `KVCacheSpec.merge()` is called, it does `deepcopy(specs[0])`. If the first spec in a group is GDN (2 shapes), ALL layers in that group — including KDA — get only 2 state tensors. KDA then tries to unpack 4 values from 2:

```python
# KDA forward:
(conv_state_q, conv_state_k, conv_state_v, recurrent_state) = constant_caches
#  ^^^^^^^^^^   ^^^^^^^^^^   ^^^^^^^^^^   ^^^^^^^^^^^^^^^^
#  Only 2 values available — CRASH!
```

### The lesson

**`_reshape_kv_cache_tensors` uses the GROUP spec (from `merge`) for ALL layers in a group.** This means layers in the same group MUST have identical `shapes` and `dtypes`, not just identical block-table parameters. The UnifiedMambaSpec approach is fundamentally unsafe because it allows structurally incompatible layers to be merged into a single group.

---

## The Actual Fix: Monkey-Patching the Grouping Function

Since we can't make GDN and KDA share a group (their shapes are structurally incompatible), we fix the problem at the source: the `group_size` computation. We keep GDN and KDA in separate groups but ensure the group_size doesn't degenerate to 1.

### The monkey-patch

```python
def _patch_kv_cache_grouping() -> None:
    import vllm.v1.core.kv_cache_utils as _kcu
    _original = _kcu._get_kv_cache_groups_uniform_page_size

    def _patched(kv_cache_spec: dict) -> list:
        from collections import defaultdict
        from math import ceil as _ceil

        same_type_layers: defaultdict[object, list[str]] = defaultdict(list)
        for layer_name, layer_spec in kv_cache_spec.items():
            same_type_layers[layer_spec].append(layer_name)

        min_num = min(len(v) for v in same_type_layers.values())
        max_num = max(len(v) for v in same_type_layers.values())

        if min_num > 2 or max_num < min_num * 1.25:
            # Not a singleton/near-singleton case — use original logic.
            return _original(kv_cache_spec)

        # Singleton / near-singleton type detected.
        # Use max_num as group_size to produce at most num_types groups.
        group_size = max_num

        grouped_layers = []
        for layers in same_type_layers.values():
            num_groups = _ceil(len(layers) / group_size)
            for i in range(num_groups):
                grouped_layers.append(layers[i::num_groups])
        return _kcu.create_kv_cache_group_specs(kv_cache_spec, grouped_layers)

    _kcu._get_kv_cache_groups_uniform_page_size = _patched

_patch_kv_cache_grouping()
```

### How it works

The patch detects when the original algorithm would produce a degenerate `group_size` (min <= 2 with large max/min ratio) and overrides it with `max_num_layers`.

### Effect on `a12_g1_k11`

```
same_type_layers (unchanged — GDN and KDA stay separate):
  FullAttentionSpec:   12 layers
  MambaSpec(2-tuple):  1 layer   (GDN)
  MambaSpec(4-tuple):  11 layers (KDA)

min = 1, max = 12
Patch activates: min_num(1) <= 2 AND max_num(12) >= min_num * 1.25

group_size = 12 (max)

Groups:
  Attention: [attn_0 ... attn_11]   -- 1 group of 12
  GDN:       [gdn_0]                -- 1 group of 1 (with 11 unused positions)
  KDA:       [kda_0 ... kda_10]     -- 1 group of 11

Total: 3 groups  (was 24!)
Tensors: 12 (each shared across all groups that have a layer at that position)
```

3 metadata builds per forward instead of 24. Each group has the correct spec type — GDN group uses GDN's 2-tuple shapes, KDA group uses KDA's 4-tuple shapes.

### Why this fix is safe

1. **Each group contains only one spec type.** GDN and KDA are in separate groups, so `merge()` always merges identical specs. No shape mismatch.
2. **`group_size` controls tensor sharing, not correctness.** With `group_size = 12`, there are 12 physical KV cache tensors. The GDN group's 1 layer uses tensor position 0. The KDA group's 11 layers use positions 0-10. The attention group uses all 12. Unused positions are simply idle memory — but this "wasted" memory is minimal because `page_size` is already unified.
3. **`num_blocks` is computed correctly.** `num_blocks = available_memory // page_size // group_size`. A larger `group_size` means fewer blocks per tensor, which is the correct tradeoff (fewer groups = less metadata overhead vs. slightly less KV cache capacity).
4. **Falls back to original logic for well-balanced models.** The patch only activates when `min_num <= 2`, so models like Gemma3 (designed for the n:1 pattern) are unaffected.

---

## The Sliding Window Case

The same grouping problem affects attention layers with different window configurations. A model like `a9_g14_s1` (9 full attention + 14 GDN + 1 sliding window attention) creates a singleton sliding-window type that degenerates `group_size` to 1.

But there's a second, more immediate problem: **FlashInfer requires all layers in a group to have the same `window_left`**. The check happens in `infer_global_hyperparameters()` (`vllm/v1/attention/backends/utils.py`), which reads `impl.sliding_window` from each FlashInfer attention implementation in the group. If values differ, it raises:

```
ValueError: Window left is not the same for all layers
```

This means even if the grouping were fixed, mixing sliding-window and full-attention layers in the same group would crash during CUDA graph capture.

### Fix: Return `SlidingWindowSpec` for sliding-window layers

```python
def get_kv_cache_spec(self, vllm_config):
    config = vllm_config.model_config.hf_config
    block_size, _ = get_unified_page_size_for_config(config, vllm_config)
    # ... dtype resolution ...

    if self.window_size is not None:
        return SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            dtype=kv_cache_dtype,
            sliding_window=self.window_size,
        )

    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=self.num_kv_heads,
        head_size=self.head_dim,
        dtype=kv_cache_dtype,
    )
```

This puts sliding-window layers in a separate KV cache group from full-attention layers, so each group has homogeneous `window_left` values. The monkey-patched grouping function handles the resulting singleton group without degenerating `group_size`.

### Why not always use `FullAttentionSpec`?

An earlier version of the fix returned `FullAttentionSpec` for all layers regardless of window size. This avoided the grouping issue but was incorrect: it told vLLM the layer has no sliding window, while the `Attention` implementation still had `per_layer_sliding_window` set. The FlashInfer backend reads `impl.sliding_window` directly from the attention implementation (not from the KVCacheSpec), so layers in the same group would still have mismatched `window_left` values and crash.

The correct fix is to let the spec reflect reality (`SlidingWindowSpec` for windowed layers) and let the grouping monkey-patch handle the singleton.

---

## Runtime Crash: Missing PIECEWISE Splitting Op

After applying the grouping monkey-patch, benchmarks crashed with:

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

at `fused_gdn_gating_kernel` during prefill. The crash only occurred mid-benchmark (not on the first forward pass), which pointed to a CUDA graph replay issue.

### Root cause: `vllm::apriel2_gdn_attention_core` was not a splitting op

vLLM's PIECEWISE CUDA graph mode captures the model forward in **pieces**, splitting at custom ops listed in `CompilationConfig._attention_ops`. Each such op becomes a graph break — the compiled code stops the current piece, dispatches to the op at runtime, then starts the next piece.

If a custom op is NOT in `_attention_ops`, it gets **baked into** a compiled piece. This means:

- The op's tensor addresses and shapes are frozen at capture time
- Replaying the piece with different batch sizes (prefill vs decode) accesses wrong memory → illegal memory access

vLLM's default `_attention_ops` list (`vllm/config/compilation.py:610-622`) includes `vllm::gdn_attention_core` (from Qwen3-Next) and `vllm::kda_attention`, but **NOT** `vllm::apriel2_gdn_attention_core`. Our Apriel2-specific GDN op has a different name because it uses a different kernel implementation.

### Why this only matters in `FULL_AND_PIECEWISE` mode

In `FULL_AND_PIECEWISE` mode (the default for non-stochastic models):

- **FULL graphs** handle pure-decode batches — the entire forward is one graph, no splitting needed
- **PIECEWISE graphs** handle prefill and mixed prefill+decode batches — these NEED the graph breaks

The GDN op registration was originally inside an `if not APRIEL2_FULL_CUDA_GRAPHS:` guard, because it was only thought necessary for stochastic (non-FULL) mode. But `APRIEL2_FULL_CUDA_GRAPHS` defaults to `True`, which means `FULL_AND_PIECEWISE` mode — where PIECEWISE graphs ARE active for prefill.

### Fix

Register `vllm::apriel2_gdn_attention_core` **unconditionally** (outside the FULL guard):

```python
# The GDN op MUST always be registered: even in FULL_AND_PIECEWISE mode,
# PIECEWISE graphs handle prefill.
_gdn_op = "vllm::apriel2_gdn_attention_core"
if _gdn_op not in CompilationConfig._attention_ops:
    CompilationConfig._attention_ops.append(_gdn_op)

# Stochastic dispatch op only needed when FULL graphs are disabled.
if not APRIEL2_FULL_CUDA_GRAPHS:
    _stochastic_op = "vllm::stochastic_mixer_dispatch"
    if _stochastic_op not in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops.append(_stochastic_op)
```

### Diagnostic tip

To verify splitting ops are registered, check the vLLM startup log for:

```
Added vllm::apriel2_gdn_attention_core to vLLM splitting ops
```

Or dump `CompilationConfig._attention_ops` at startup. If the Apriel2 GDN op is missing, prefill batches will crash with illegal memory access on the first GDN layer.

---

## Runtime Crash: Triton constexpr JIT on Large Prefill Batches

Even after fixing the splitting op registration, the same illegal memory access persisted at `fused_gdn_gating_kernel`. Debug sync points proved the crash was NOT in the attention kernel or `causal_conv1d_fn` — it was specifically in the Triton gating kernel.

### Root cause: `total_elements: tl.constexpr`

The `fused_gdn_gating_kernel` Triton kernel originally declared `total_elements` as `tl.constexpr`:

```python
@triton.jit
def fused_gdn_gating_kernel(
    ...,
    total_elements: tl.constexpr,  # ← PROBLEM
    BLOCK_SIZE: tl.constexpr,
    ...
):
```

`tl.constexpr` tells Triton to **bake the value into the compiled kernel binary**. Different values require a completely new JIT compilation — Triton generates a new PTX binary, loads it via `cuModuleLoadData`, and caches it by the constexpr key.

During benchmarking:

1. Decode batches have small `total_elements` (e.g., 128, 256) — these kernels compile and cache fine
2. Prefill batches have large `total_elements` (e.g., 16384) — this triggers a new JIT compilation
3. The `load_binary` call during CUDA graph capture/replay causes the illegal memory access

The exact failure mechanism: Triton's JIT compilation calls `torch.cuda.synchronize()` internally during binary loading. If this happens during CUDA graph capture (where `synchronize()` is forbidden), or if the newly loaded binary's function pointer differs from what was captured, the graph replays stale pointers → illegal memory access.

### Fix

Change `total_elements` from `tl.constexpr` to a regular kernel argument:

```python
@triton.jit
def fused_gdn_gating_kernel(
    ...,
    total_elements,  # Regular arg — no JIT recompilation per batch size
    BLOCK_SIZE: tl.constexpr,
    ...
):
```

As a regular argument, `total_elements` is passed as a kernel launch parameter, not baked into the binary. The same compiled kernel handles all batch sizes. The `mask = offset < total_elements` check still works correctly — Triton just evaluates it at runtime instead of compile time.

### When to use `tl.constexpr` vs regular args

| Use `tl.constexpr` | Use regular arg |
|---------------------|-----------------|
| Block sizes, tile shapes (affects register allocation) | Tensor dimensions that vary per call |
| Number of heads (affects loop unrolling) | Batch-dependent sizes like `total_elements` |
| Threshold constants | Any value that changes between prefill and decode |

**Rule of thumb**: If a value can differ between CUDA graph capture and replay, it MUST NOT be `tl.constexpr`.

---

## Runtime Crash: Import-by-Value Bug in cudagraph_capturing_enabled

After fixing the three issues above (grouping, splitting op, constexpr), the benchmark crashed during large re-prefill after KV cache preemption:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

at `initial_state[~has_initial_state, ...] = 0` in the GDN layer's `_forward_core()`. The crash only appeared when vLLM evicted requests and re-prefilled them with large token batches (e.g., 16384 tokens).

### Debugging odyssey

Adding `torch.cuda.synchronize()` debug sync points proved difficult because they were gated by:

```python
if DEBUG_SYNC and not torch.compiler.is_compiling() and not cudagraph_capturing_enabled:
```

The syncs **never fired**. Three runs with different debug approaches all showed no sync output, yet the crash persisted at the same location. The imported `cudagraph_capturing_enabled` was always `True`.

### Root cause: Python import-by-value semantics

```python
# In vllm/compilation/monitor.py:
cudagraph_capturing_enabled: bool = True  # <-- default is True!

def set_cudagraph_capturing_enabled(enabled: bool) -> None:
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled
```

```python
# In modeling_apriel2.py:
from vllm.compilation.monitor import cudagraph_capturing_enabled
```

Python's `from module import name` copies the **value** at import time. When vLLM later calls `set_cudagraph_capturing_enabled(False)`, the module's global changes but the imported name in `modeling_apriel2.py` still holds the original `True`.

This caused two critical failures:

**1. Debug syncs were permanently disabled** — all `not cudagraph_capturing_enabled` guards evaluated to `not True` = `False`, silently skipping every sync point. This wasted three debugging iterations trying to figure out why sync output never appeared.

**2. `num_actual_tokens` was always clamped to `num_cache_lines`** — the guard intended to run only during CUDA graph capture ran unconditionally:

```python
# Line 1936 (GDN) and 2423 (KDA):
if cudagraph_capturing_enabled and num_actual_tokens > num_cache_lines:
    num_actual_tokens = num_cache_lines  # Always executes!
```

The clamping was designed for CUDA graph capture, where the dummy batch size can exceed the number of allocated cache lines. During normal inference, `num_actual_tokens` should NOT be clamped — the full token batch must be processed.

### The crash mechanism

During a 16384-token re-prefill after preemption:

1. `num_cache_lines = 1606` (KV cache blocks for GDN layer)
2. `num_actual_tokens = 16384` (the full re-prefill batch)
3. The always-True guard clamps `num_actual_tokens` to 1606
4. `mixed_qkv = mixed_qkv[:1606]` — input tensor truncated to 1606 tokens
5. But `query_start_loc = [0, 15873, 16384]` — still references all 16384 tokens
6. `causal_conv1d_fn` Triton kernel reads positions 1606–16383 from the truncated tensor → **illegal memory access**

The diagnostic output confirmed the mismatch:

```
[PRE-CONV] model.layers.12.mixer: conv_state.shape=torch.Size([1606, 1152, 3]),
  x.shape=torch.Size([1152, 1606]),       ← truncated!
  query_start_loc=[0, 15873, 16384],      ← still full size!
```

### Fix

Import the module and access the attribute dynamically:

```python
import vllm.compilation.monitor as _compile_monitor

# Before (broken — always True):
if cudagraph_capturing_enabled and num_actual_tokens > num_cache_lines:

# After (reads live value):
if _compile_monitor.cudagraph_capturing_enabled and num_actual_tokens > num_cache_lines:
```

All references to `cudagraph_capturing_enabled` in `modeling_apriel2.py` were updated to use `_compile_monitor.cudagraph_capturing_enabled`.

### After the fix

```
[PRE-CONV] model.layers.12.mixer: conv_state.shape=torch.Size([1606, 1152, 3]),
  x.shape=torch.Size([1152, 16384]),      ← full size, not truncated!
  query_start_loc=[0, 15873, 16384],
[SYNC-2] model.layers.12.mixer: post-causal_conv1d_fn sync OK
[SYNC-3] model.layers.12.mixer: post-fused_gdn_gating sync OK
[SYNC-5] model.layers.12.mixer: post-chunk_gated_delta_rule sync OK
```

Multiple re-prefills succeeded without any crashes.

---

## Key Takeaways and Learnings

1. **vLLM's grouping algorithm was designed for fixed-pattern hybrids** (like Gemma3's 5:1 ratio), not arbitrary supernet placements. The `group_size = min(layers per type)` heuristic degenerates when any type has a singleton.

2. **Metadata overhead is per-group, not per-layer**. On small models where decode compute is fast, the number of groups can dominate total forward time. This is a quadratic relationship: more groups means more metadata builds, each of which adds a fixed ~0.5ms cost.

3. **`_reshape_kv_cache_tensors` uses the GROUP spec, not per-layer specs.** This is the most important lesson. Although `initialize_kv_cache_tensors` has access to per-layer specs, `_reshape_kv_cache_tensors` iterates over groups and uses `group.kv_cache_spec` (from `merge()` = `deepcopy(specs[0])`) for ALL layers in the group. Layers within a group MUST have identical `shapes`/`dtypes`, not just compatible block-table parameters.

4. **Custom equality on frozen dataclasses is dangerous when `merge()` does `deepcopy(specs[0])`.** Making two specs "equal" via custom `__eq__` doesn't mean they're interchangeable — `merge()` picks one arbitrarily and uses its full structure. If the "losing" spec has different shapes, those shapes are silently lost.

5. **Monkey-patching the grouping function is safer than subclassing the spec.** The spec equality approach tries to lie about structural compatibility. The grouping patch keeps specs honest and just fixes the `group_size` computation. Each group still contains only genuinely identical specs.

6. **Impact**: For `a12_g1_k11`, the fix reduces groups from 24 to 3, eliminating ~10ms of metadata overhead per decode step and restoring throughput to the expected level.

7. **Always test with the exact failing config.** The UnifiedMambaSpec approach seemed correct in analysis but crashed immediately on the `a12_g1_k11` layout because we missed the `_reshape_kv_cache_tensors` code path that uses group specs.

8. **Custom ops with different names need separate splitting op registration.** vLLM's `_attention_ops` list is name-based. `vllm::gdn_attention_core` (Qwen3-Next) and `vllm::apriel2_gdn_attention_core` (Apriel2) are different ops. Missing registration → op gets baked into compiled PIECEWISE graph pieces → illegal memory access on batch size changes.

9. **PIECEWISE graphs are active even in `FULL_AND_PIECEWISE` mode.** FULL handles decode, PIECEWISE handles prefill. Any fix that only applies in non-FULL mode will miss prefill crashes.

10. **Never use `tl.constexpr` for batch-dependent kernel arguments.** Triton bakes constexpr values into the compiled binary. Different values trigger JIT recompilation, which can call `synchronize()` during graph capture or produce stale function pointers during replay. Use `tl.constexpr` only for structural parameters (block sizes, head counts, thresholds) that never change between capture and replay.

11. **FlashInfer requires homogeneous `window_left` within a group.** `infer_global_hyperparameters()` checks `impl.sliding_window` from the actual attention implementation, not the KVCacheSpec. Layers with different window sizes MUST be in separate groups, which means returning different spec types (`SlidingWindowSpec` vs `FullAttentionSpec`).

12. **Never use `from module import mutable_global` for flags that change at runtime.** Python's `from module import name` copies the value at import time. If the module later mutates the global (e.g., `set_cudagraph_capturing_enabled(False)`), the imported name retains the stale value. Use `import module; module.flag` instead. This bug silently disabled all debug sync points AND caused the `num_actual_tokens` clamping guard to fire unconditionally, truncating prefill input tensors while `query_start_loc` still referenced the full batch → OOB memory access in `causal_conv1d_fn`.

---

## Key Code Locations

| What | Where |
|------|-------|
| Grouping algorithm | `vllm/v1/core/kv_cache_utils.py` : `_get_kv_cache_groups_uniform_page_size()` |
| group_size computation | Same file, lines 1033-1042 |
| KVCacheTensor creation | Same file, `get_kv_cache_config_from_groups()` |
| num_blocks formula | Same file, `get_num_blocks()`: `available_memory // page_size // num_layers` |
| Per-forward metadata loop | `vllm/v1/worker/gpu_model_runner.py` : the nested `for kv_cache_gid / for attn_gid` loop |
| MambaSpec definition | `vllm/v1/kv_cache_interface.py` : `class MambaSpec(KVCacheSpec)` |
| Monkey-patch fix | `fast_llm_external_models/apriel2/vllm/modeling_apriel2.py` : `_patch_kv_cache_grouping()` |
| `_reshape_kv_cache_tensors` (group spec!) | `vllm/v1/worker/gpu_model_runner.py` : uses `group.kv_cache_spec` for all layers |
| GDN get_kv_cache_spec | Same file, `Apriel2GatedDeltaNet.get_kv_cache_spec()` |
| KDA get_kv_cache_spec | Same file, `Apriel2KDAMixer.get_kv_cache_spec()` |
| Attention get_kv_cache_spec (SlidingWindowSpec) | Same file, `Apriel2Attention.get_kv_cache_spec()` |
| Cache initialization (per-layer shapes) | `vllm/v1/worker/gpu_model_runner.py` : `initialize_kv_cache_tensors()` |
| merge assertion | `vllm/v1/kv_cache_interface.py` : `KVCacheSpec.merge()` |
| PIECEWISE splitting op registration | Same file, after `stochastic_mixer_dispatch` registration |
| Default `_attention_ops` list | `vllm/config/compilation.py` : lines 610-622 |
| `fused_gdn_gating_kernel` (Triton) | Same file, `@triton.jit` kernel |
| FlashInfer `window_left` check | `vllm/v1/attention/backends/utils.py` : `infer_global_hyperparameters()` |
| `SlidingWindowSpec` definition | `vllm/v1/kv_cache_interface.py` : `class SlidingWindowSpec(AttentionSpec)` |
| `cudagraph_capturing_enabled` default | `vllm/compilation/monitor.py` : line 45, defaults to `True` |
| `set_cudagraph_capturing_enabled()` | `vllm/compilation/monitor.py` : mutates module global |
| Import fix (`_compile_monitor`) | `modeling_apriel2.py` : `import vllm.compilation.monitor as _compile_monitor` |
| `num_actual_tokens` clamping (GDN) | `modeling_apriel2.py` : `_forward_core()`, `if _compile_monitor.cudagraph_capturing_enabled` |
| `num_actual_tokens` clamping (KDA) | `modeling_apriel2.py` : `Apriel2KDAMixer._forward_core()`, same pattern |
| Debug sync points (`APRIEL2_DEBUG_SYNC`) | `modeling_apriel2.py` : SYNC-1 through SYNC-6, gated by `DEBUG_SYNC` flag |
