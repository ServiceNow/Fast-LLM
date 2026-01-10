# Apriel2 Conversion System: Algebraic Structure

This document describes the algebraic structure underlying the Apriel2 conversion
and surgery system, including its mathematical properties and practical limitations.

## Overview

The conversion system transforms model weights between architectures using a
**declarative, plan-based approach**. The key insight is separating:

1. **Config composition**: What the target architecture looks like
2. **Plan building**: How to transform weights to get there
3. **Plan execution**: Actually performing the transformation

Each layer has its own algebraic structure with specific guarantees and limitations.

---

## Conceptual Types

The system operates on three conceptual types (all `dict` at runtime):

| Type | Description | Has `init` field? | Example |
|------|-------------|-------------------|---------|
| **S (State)** | Complete model config | No | A saved `config.json` |
| **P (Partial Surgery)** | Incomplete config specifying changes | May have | `{"decoder": {"block": {"mixer": {"type": "gdn"}}}}` |
| **T (Transition Spec)** | Complete config with init metadata | Yes | Result of `compose_configs(S, P)` |

The `init` field controls weight initialization:
- `init: transfer` → Use weight conversion (MIL, DIL, KIL, or passthrough)
- `init: random` → Randomly initialize weights

---

## Layer 1: Config Composition

### The Surgery Monoid (P, ∘, {})

Partial surgeries form a **monoid** under deep merge:

```
compose_configs : P × P → P     (deep merge, overlay wins)
```

**Properties:**
- **Identity**: `compose_configs(p, {}) = compose_configs({}, p) = p`
- **Associativity**: `compose_configs(compose_configs(a, b), c) = compose_configs(a, compose_configs(b, c))`

This is a **total operation** - it always succeeds.

### Surgery Action on States

Surgeries act on states to produce transition specs:

```
compose_configs : S × P → T     (apply surgery with inheritance)
compose_configs : T × P → T     (extend transition spec)
```

This is also a **total operation** - config composition never fails.

### The Action Law (Conditional)

For the action to be a proper monoid action, we need:

```
(s · p₁) · p₂ = s · (p₁ ∘ p₂)
```

**This law holds ONLY for additive surgeries.**

| Surgery Type | Example | Action Law |
|--------------|---------|------------|
| **Additive** | Adding to `mixers` dict without changing outer `type` | ✓ Holds |
| **Replacement** | Declaring `type: mamba` to replace `type: attention` | ✗ Violated |

For replacement surgeries, the system uses **last-write-wins** semantics:
- `p₁ ∘ p₂` produces `p₂`'s type (overlay wins)
- `(s · p₁) · p₂` goes through `p₁`'s type as intermediate state

**Example of action law violation:**

```python
s  = attention config
p1 = {"decoder": {"block": {"mixer": {"type": "mamba", ...}}}}
p2 = {"decoder": {"block": {"mixer": {"type": "attention", ...}}}}

# Sequential: goes through mamba, loses attention geometry
(s · p1) · p2  →  attention config (minimal, lost head_groups/head_size)

# Merged: skips mamba entirely
s · (p1 ∘ p2)  →  attention config (preserved geometry from s)
```

---

## Layer 2: Plan Building

### Plan Building is a Partial Function

```
plan_surgery : S × T → Plan    (may fail!)
```

Plan building can fail when:
1. `init: transfer` is specified but no converter exists for the type pair
2. Required geometry information is missing

**Available converters (one-way only):**

| Source | Target | Converter |
|--------|--------|-----------|
| attention | mamba | MIL (Mamba Initialization from LLM) |
| attention | gdn | DIL (Delta-net Initialization from LLM) |
| attention | kda | KIL (Kimi Initialization from LLM) |
| attention | attention | Passthrough (same-type) |
| any | any | Random init (if `init: random`) |

**No reverse converters exist.** You cannot do `mamba → attention` with `init: transfer`.

### Plan Composition

```
compose : Plan(A→B) × Plan(B→C) → Plan(A→C)
```

Plan composition is:
- **Total**: Always succeeds (just substitutes Ref expressions)
- **Associative**: `(P₁ ∘ P₂) ∘ P₃ = P₁ ∘ (P₂ ∘ P₃)`

Plan composition does **not** perform algebraic simplification. If you had:
```
Plan1: x → MIL(x)           (attention → mamba)
Plan2: y → REVERSE_MIL(y)   (hypothetical mamba → attention)
```

Composition would give `x → REVERSE_MIL(MIL(x))`, not `x → x`. The expressions
are substituted, not simplified.

### Functoriality (Conditional)

When all intermediate plans can be built:

```
compose(plan(S₀,T₁), plan(T₁,T₂), ...) ≡ plan(S₀, Tₙ)
```

where `≡` denotes semantic equivalence (identical weights when executed).

**This only holds when all `plan(Tᵢ, Tᵢ₊₁)` calls succeed.**

---

## Layer 3: The Full Pipeline

### build_plan Behavior

The `build_plan` function in `convert.py` applies surgeries **sequentially**:

```python
for surgery_config in surgery_configs:
    target_config = compose_configs(current_config, surgery_config)  # S × P → T
    surgery_plan = plan_surgery(current_config, target_config)       # May fail!
    current_plan = compose(current_plan, surgery_plan)
    current_config = strip_init_fields(target_config)                # T → S
```

Each surgery is applied one at a time, and plan building happens in the loop.

### Sequential vs Merged Application

This creates an important behavioral difference:

| Approach | Config Path | Plan Building | Result |
|----------|-------------|---------------|--------|
| **Sequential** | `s → mamba → attention` | Fails at step 2 | Error |
| **Merged** | `s → attention` (mamba skipped) | Succeeds | No-op |

**Example:**

```python
# Surgery 1: attention → mamba
p1 = {"decoder": {"block": {"mixer": {"type": "mamba", "init": "transfer", ...}}}}

# Surgery 2: mamba → attention
p2 = {"decoder": {"block": {"mixer": {"type": "attention", "init": "transfer", ...}}}}

# Sequential (current build_plan behavior):
# Step 1: plan_surgery(attention, mamba) → MIL plan ✓
# Step 2: plan_surgery(mamba, attention) → ERROR: No converter!

# If surgeries were merged first:
merged = compose_configs(p1, p2)  # Results in attention surgery (overlay wins)
# plan_surgery(attention, attention) → Passthrough plan ✓ (no-op)
```

### Design Rationale

The sequential approach is intentional:

1. **Explicit lossy operations**: Forces users to acknowledge when weights can't be transferred
2. **Catches mistakes**: If you write `mamba → attention` with `init: transfer`, you probably made an error
3. **No surprising no-ops**: The merged approach would silently produce identity, hiding the round-trip

If you want to go `attention → mamba → attention`:
- First step: `init: transfer` (uses MIL)
- Second step: `init: random` (can't recover original attention weights)

---

## Summary: What's Total vs Partial

| Operation | Total/Partial | Failure Mode |
|-----------|---------------|--------------|
| `compose_configs(P, P)` | **Total** | Never fails |
| `compose_configs(S, P)` | **Total** | Never fails |
| `plan_surgery(S, T)` | **Partial** | No converter for type pair |
| `compose(Plan, Plan)` | **Total** | Never fails |
| `execute(Plan, weights)` | **Total** | Never fails (given valid plan) |

## Summary: What Laws Hold Where

| Law | Scope | Holds? |
|-----|-------|--------|
| Surgery monoid (associativity) | All surgeries | ✓ Always |
| Action law `(s·p₁)·p₂ = s·(p₁∘p₂)` | Additive surgeries only | ✓ Conditional |
| Plan composition associativity | All plans | ✓ Always |
| Functoriality | When all intermediate plans build | ✓ Conditional |

---

## Practical Guidelines

### Additive Surgery Patterns (Safe)

These patterns preserve the action law and always build:

```yaml
# Wrap in stochastic (keeps original mixer inside)
decoder:
  block:
    mixer:
      type: stochastic
      main_mixer_name: attention
      mixers:
        attention: {init: transfer}

# Add sub-mixer to existing stochastic
decoder:
  block:
    mixer:
      mixers:
        new_mixer: {type: gdn, init: transfer, ...}

# Modify parameters without changing type
decoder:
  block:
    mixer:
      window_size: 512
```

### Replacement Surgery Patterns (Use with Care)

These patterns violate the action law and may fail plan building:

```yaml
# Type replacement - action law violated
decoder:
  block:
    mixer:
      type: mamba  # Replaces attention
      init: transfer

# Reverse conversion - plan building fails
decoder:
  block:
    mixer:
      type: attention  # From mamba source
      init: transfer   # ERROR: no converter

# Reverse conversion - must use random init
decoder:
  block:
    mixer:
      type: attention
      init: random     # OK: randomly initialize
      heads: 8
      head_groups: 4
      head_size: 32
```

### Debugging Tips

1. **Use `--dry-run`** to see the plan without executing:
   ```bash
   python convert.py input output -s surgery.yaml --dry-run
   ```

2. **Use `--show-plan`** to visualize the expression tree:
   ```bash
   python convert.py input output -s surgery.yaml --show-plan
   ```

3. **Check for `init: transfer` on reverse conversions** - this is the most common
   source of "No converter available" errors.

---

## Supernet Creation (Stochastic Wrapping)

A "supernet" is a model where each layer has multiple mixer options via a stochastic
mixer. During training, the model samples which mixer to use, enabling neural
architecture search or mixture-of-experts style training.

### Creating a Supernet from a Base Model

See `examples/stochastic_supernet.yaml` for a complete example.

```bash
# Convert attention model to supernet with 4 mixer types
python convert.py base_checkpoint output/ \
    -s examples/stochastic_supernet.yaml
```

### Example Surgery

```yaml
decoder:
  block:
    mixer:
      type: stochastic
      main_mixer_name: attention
      sampling_strategy: uniform
      mixers:
        # Attention - direct weight transfer
        attention:
          type: attention
          init: transfer

        # Sliding window - transfer with window size
        sliding_window:
          type: attention
          init: transfer
          window_size: 4096

        # GDN - DIL initialization from attention
        gdn:
          type: gdn
          init: transfer
          convolution_layer:
            kernel_size: 4

        # KDA - KIL initialization from attention
        kda:
          type: kda
          init: transfer
          convolution_layer:
            kernel_size: 4

    mlp:
      init: transfer
    normalization:
      init: transfer
```

### Weight Initialization

When creating a supernet from a non-stochastic source:

| Sub-mixer | Source | Initialization |
|-----------|--------|----------------|
| `attention` | attention | Passthrough (same type) |
| `sliding_window` | attention | Passthrough (attention variant) |
| `gdn` | attention | DIL conversion |
| `kda` | attention | KIL conversion |
| `mamba` | attention | MIL conversion |

The `main_mixer_name` specifies which sub-mixer is the "primary" one. This affects:
- Which mixer is used for inference by default
- Which sub-mixer provides weights when unwrapping (see Supernet Pruning below)

---

## Supernet Pruning (Stochastic Unwrapping)

A common use case is pruning a "supernet" (stochastic mixer with multiple sub-mixers)
to a heterogeneous network where each layer uses a single mixer type.

### The Challenge

When unwrapping `stochastic → non-stochastic`, the system uses `main_mixer_name`
as the weight source. If your supernet has `main_mixer_name: attention` but you
want to extract the `gdn` sub-mixer, a naive surgery would use DIL conversion
from attention instead of preserving the existing gdn weights.

### The Solution: Two-Step Surgery

Use two surgeries in sequence:

1. **Step 1**: Set `main_mixer_name` per block type (config-only, all weights passthrough)
2. **Step 2**: Unwrap to non-stochastic (weights come from the correct sub-mixer)

### Example

See `examples/prune_supernet_step1.yaml` and `examples/prune_supernet_step2.yaml`.

```bash
# Prune a homogeneous supernet to heterogeneous [attn, gdn, kda, swa] pattern
python convert.py supernet_checkpoint output/ \
    -s examples/prune_supernet_step1.yaml \
    -s examples/prune_supernet_step2.yaml
```

**Step 1** converts fixed → pattern and sets different `main_mixer_name` per block:

```yaml
decoder:
  type: pattern
  pattern: [attn_block, gdn_block, kda_block, swa_block]
  blocks:
    attn_block:
      mixer: {main_mixer_name: attention}
    gdn_block:
      mixer: {main_mixer_name: gdn}
    kda_block:
      mixer: {main_mixer_name: kda}
    swa_block:
      mixer: {main_mixer_name: sliding_window}
```

**Step 2** unwraps each block to its main mixer type:

```yaml
decoder:
  blocks:
    attn_block:
      mixer: {type: attention, init: transfer}
    gdn_block:
      mixer: {type: gdn, init: transfer, convolution_layer: {kernel_size: 4}}
    kda_block:
      mixer: {type: kda, init: transfer, convolution_layer: {kernel_size: 4}}
    swa_block:
      mixer: {type: attention, init: transfer, window_size: 4096}
```

### Why This Works

- Step 1 is config-only (all Ref expressions = passthrough)
- Step 2 uses `main_mixer_name` to find the source, which now points to the correct sub-mixer
- Each layer extracts weights from its designated sub-mixer, not from attention via conversion

---

## References

- `config.py`: Config composition implementation and detailed docstrings
- `expr.py`: Expression types and plan composition
- `converters.py`: MIL, DIL, KIL converter implementations
- `test_plan_execution.py`: Algebraic law tests
- `test_conversion_e2e.py`: End-to-end pipeline tests
