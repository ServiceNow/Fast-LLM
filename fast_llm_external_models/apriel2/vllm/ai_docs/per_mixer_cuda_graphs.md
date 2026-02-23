# Per-Mixer CUDA Graph Caching for Stochastic Mixers

## Problem

Apriel2's supernet uses `Apriel2StochasticMixer` — a wrapper that routes each layer to one of several mixer types (attention, GDN, KDA, sliding_window) based on the active placement. The `stochastic_mixer_dispatch` custom op is registered as a CUDA graph splitting op, forcing vLLM into **PIECEWISE** mode.

In PIECEWISE mode, vLLM captures CUDA graphs for the compute pieces between split points (norms, MLPs) but runs the split points themselves eagerly. This means every mixer forward at every layer incurs full kernel launch overhead on every decode step.

**Measured impact**: PIECEWISE mode achieves ~290 tok/s, while FULL CUDA graph mode with a fixed layout (no supernet) achieves ~900 tok/s — a **3x gap** from the split points running eagerly.

## Approach

Cache a separate `torch.cuda.CUDAGraph` per (mixer_name, num_tokens) at each stochastic layer. During decode, replay the cached graph instead of running the mixer eagerly.

### Implementation (in `modeling_apriel2.py`)

- **`APRIEL2_MIXER_CUDA_GRAPHS`** env var — gates the feature (default `"0"`)
- **`MixerGraphEntry`** — dataclass holding a captured graph + input/output pointer addresses
- **`MixerGraphCache`** — per-layer cache keyed by `(mixer_name, num_tokens)`
- **`_capture_all_mixers_for_num_tokens()`** — captures graphs during `capture_model()` with eager warmup before each capture (for Triton autotuning)
- **`_batch_has_prefill()`** — detects mixed prefill-decode batches that can't use graph replay
- **`stochastic_mixer_dispatch`** modified with capture/replay/eager-fallback logic
- Cache instance stored as `Apriel2StochasticMixer._mixer_graph_cache`

### Dispatch Flow

```text
stochastic_mixer_dispatch(hidden_states, output, positions, layer_name):
  if cache is not None:
    if capturing and runtime_mode == PIECEWISE:
      → capture graphs for all/active mixers, return
    if not prefill_batch and cache.has(active_mixer, num_tokens):
      → cache.replay(), return
  → eager fallback: active_mixer(hidden_states, output, positions)
```

## Bugs Encountered & Fixed

### 1. CUBLAS_STATUS_NOT_INITIALIZED during profile_run

`cudagraph_capturing_enabled` defaults to `True` in `vllm.compilation.monitor`. During `profile_run()` (before `capture_model()`), our code tried to capture graphs, but cuBLAS wasn't initialized yet.

**Fix**: Gate capture on `runtime_mode != CUDAGraphMode.NONE` (NONE during profile_run, PIECEWISE during capture_model).

### 2. Triton autotuning inside graph capture

KDA's `fused_kda_gate` uses `@triton.autotune`. First call triggers benchmarking with `cuda.synchronize()` — illegal during stream capture.

**Fix**: Run each mixer eagerly once before capturing (warmup triggers autotuning outside capture context).

### 3. GPU memory pressure from too many captured graphs (CRITICAL)

Capturing graphs for all mixers at all batch sizes creates ~5,040 graphs (48 layers x 3 mixers x ~35 batch sizes). This causes a **2.2x throughput regression** regardless of whether graphs are replayed.

## Memory Pressure Investigation

Systematic isolation of the regression source:

| Test Configuration                          | Graphs | Warmup tok/s | vs Baseline |
| ------------------------------------------- | ------ | ------------ | ----------- |
| `CUDA_GRAPHS=0` (baseline)                  | 0      | 290          | 1.0x        |
| `CUDA_GRAPHS=1`, cache exists but empty     | 0      | 290          | 1.0x        |
| `CUDA_GRAPHS=1`, active mixer only captured | ~1,680 | 179          | 0.62x       |
| `CUDA_GRAPHS=1`, capture only (no replay)   | ~5,040 | 132          | 0.46x       |
| `CUDA_GRAPHS=1`, capture + replay           | ~5,040 | 125          | 0.43x       |
| `CUDA_GRAPHS=1`, private pool + replay      | ~5,040 | 126          | 0.43x       |

**Key findings**:

1. **Python overhead is negligible** — empty cache has zero impact (290 tok/s)
2. **Graph replay adds ~5% cost** — minimal compared to the capture overhead
3. **Private graph pool doesn't help** — total GPU memory consumption is the issue, not fragmentation of vLLM's global pool
4. **Regression scales with graph count** — 1,680 graphs = 0.62x, 5,040 = 0.43x
5. The captured graphs consume GPU memory that degrades all inference operations (likely L2 cache pressure, TLB misses, or reduced memory for temporary allocations)

## Current State

The implementation is functionally correct but the "capture everything upfront" strategy is not viable due to memory pressure. The code remains in `modeling_apriel2.py` gated behind `APRIEL2_MIXER_CUDA_GRAPHS=1` (disabled by default).

## Proposed Next Approach: Lazy Per-Placement Capture

Instead of capturing all mixers for all batch sizes during `capture_model()`:

1. **On placement set**: capture graphs only for the active mixer at each layer, only for batch sizes actually encountered
2. **On placement change**: invalidate old cache (free GPU memory), re-capture for the new placement
3. **Lazy batch sizes**: capture on first encounter of a new batch size during decode, not upfront for all 35 sizes

This would keep the graph count to ~48 (one per layer per active batch size), well within the safe memory budget.

### Open Questions

- **TP > 1 compatibility**: NCCL must be in graph-safe mode for captures involving collective ops. During `capture_model()` this is guaranteed; during inference it is not. Lazy capture may only be safe at TP=1.
- **Capture-during-inference feasibility**: Need to verify that `torch.cuda.graph()` capture works correctly when called from a piecewise split point during normal inference (not during `capture_model()`).
- **Warmup cost**: Each lazy capture requires an eager warmup (for Triton autotuning) + the capture itself. This adds latency to the first decode step after a placement change or new batch size.

## Reference: vLLM Startup Phases

```text
load_weights → profile_run() → allocate KV cache → capture_model() → inference
                 │                                      │
                 │ cudagraph_capturing=True              │ cudagraph_capturing=True
                 │ runtime_mode=NONE                     │ runtime_mode=PIECEWISE
                 │ cuBLAS NOT initialized                │ cuBLAS initialized
                 │ DO NOT capture here                   │ Safe to capture
```
