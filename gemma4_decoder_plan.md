# Gemma4 Decoder Implementation Plan (revision 2 — targets google/gemma-4-26B-A4B)

## Target checkpoint config (google/gemma-4-26B-A4B)

```jsonc
{
  "enable_moe_block": true,
  "attention_k_eq_v": true,
  "use_bidirectional_attention": "vision",
  "hidden_size_per_layer_input": 0,
  "num_kv_shared_layers": 0,
  ...
}
```

The previous revision rejected `attention_k_eq_v` and `use_bidirectional_attention="vision"` —
both of which the target checkpoint sets. Inference support for those two flags is required.

## Status of feedback (validated against actual code)

| # | Claim | Verdict |
|---|-------|---------|
| 1 | MoE is config-level, not tied to full-attention layers | **Confirmed.** `Gemma4TextDecoderLayer.__init__` always builds the MoE branch when `config.enable_moe_block=True`, regardless of `layer_type`. |
| 2 | File path is `fast_llm/layers/decoder/block.py`, not `block/decoder_block.py` | **Confirmed.** |
| 3 | New MLP must register via `dynamic_type={MLPBaseConfig: "..."}` | **Confirmed.** `MLPConfig` and `MoEMLPConfig` both use this; plain `@config_class()` would not deserialize. |
| 4 | `FieldUpdate` / `RMSNormConfig` are wrong names | **Confirmed.** Correct names: `Field(...)` (with `desc=`, `hint=`, `default=`) and `RMSNormalizationConfig`. Layer wiring uses a `layer_class` property, not a `get_layer` override. |
| 5 | `_forward` must return `(output, bias)` tuple | **Confirmed.** `BlockWithBias._forward` is abstract and returns `tuple[torch.Tensor, torch.Tensor \| None]`. New MLP must return `(combined, None)`. |
| 6 | Router scale fusion missing `hidden_size**-0.5` | **Confirmed.** `Gemma4TextRouter.forward` does `norm(x) * scale * hidden_size**-0.5` then `proj(...)`. Fused weight is `proj.weight * scale[None, :] * hidden_size**-0.5`. The `with_scale=False` RMSNorm still has to run inside `_forward`. The fusion is inference-correct but **not state-dict-roundtrip-exact**. |
| 7 | Real Fast-LLM weight prefixes are different | **Confirmed.** Actual prefixes: `decoder.{i}.mixer.{query,key_value,dense}`, `decoder.{i}.mlp.{layer_1,layer_2}`, `decoder.{i}.norm_{1,2}`, `decoder.{i}.post_mixer_normalization`, `decoder.{i}.post_mlp_normalization`, `embeddings.word_embeddings_weight`, `head.final_norm`, `head.output_weights`. |
| 8 | Use `PatternBlockSequenceConfig`, not new global fields | **Confirmed.** Apriel/Apriel2 already do this. Two block configs (sliding, full) plus a 6-element pattern is the correct mechanism. |
| 9 | Per-layer-input branch, KV-sharing, k=v all exist in HF Gemma4 | **Confirmed (target-aware).** Found `hidden_size_per_layer_input`, `per_layer_input_gate`, `per_layer_projection`, `post_per_layer_input_norm`, `layer_scalar`, `num_kv_shared_layers`, `attention_k_eq_v`. **`attention_k_eq_v` is supported via per-layer-type weight duplication** (see "Accepted with caveats" below); it is not a hard reject. The others remain hard rejects unless they're disabled (per-layer-input=0 and num_kv_shared_layers=0 in 26B A4B). |

Bonus findings (used below):
- Fast-LLM `DecoderBlock` already has `post_mixer_normalization` and `post_mlp_normalization` slots — Gemma4's outer `post_attention_layernorm` and `post_feedforward_layernorm` map directly.
- `AttentionConfig` already has `value_norm: bool` (no learnable params, `value_norm_eps` field) — exact match for Gemma's `v_norm` (`with_scale=False`).
- Fast-LLM module attribute names are `q_norm` / `k_norm` (the config field names `query_norm` / `key_norm` are different from the runtime module names; weight keys use the module name).

---

## Architecture (corrected forward)

```
residual = hidden_states
hidden_states = norm_1(residual)                         # pre_attention_layernorm
hidden_states = self_attn(hidden_states)
hidden_states = post_mixer_normalization(hidden_states)  # post_attention_layernorm
residual = residual + hidden_states

residual = hidden_states                                 # input_ in DecoderBlock.forward
hidden_states = norm_2(residual)                         # pre_feedforward_layernorm
hidden_states = mlp(hidden_states, kwargs)               # see below

# (only when enable_moe_block) mlp also runs the sparse branch internally
# and returns dense + sparse already summed

hidden_states = post_mlp_normalization(hidden_states)    # post_feedforward_layernorm
hidden_states = residual + hidden_states
```

Inside the new `Gemma4MoEMLP._forward(hidden_states, kwargs)`:

```
dense_out  = mlp_autograd(hidden_states, ...)
dense_out  = self.post_feedforward_norm_1(dense_out)

residual   = kwargs[BlockKwargs.pre_mlp_residual]        # raw pre-norm residual
flat       = residual.reshape(-1, hidden_size)
sparse_in  = self.pre_feedforward_norm_2(flat)
weights, idx = self.router(flat)                         # router uses raw residual, not norm_2 input
sparse_out = experts(sparse_in, idx, weights)
sparse_out = sparse_out.reshape_as(residual)
sparse_out = self.post_feedforward_norm_2(sparse_out)

return dense_out + sparse_out, None
```

Note: `final_logit_softcapping` and the embedding `embed_scale = sqrt(hidden_size)` factor also need
handling at converter import time but live outside the decoder block.

---

## Infrastructure changes (small, shared)

### 1. `fast_llm/layers/block/config.py` — add one kwarg key

```python
class BlockKwargs:
    ...
    pre_mlp_residual = "pre_mlp_residual"
```

### 2. `fast_llm/layers/decoder/block.py` — store residual in kwargs (and clear after use)

In `DecoderBlock.forward`, between the existing `_bias_dropout_add` call (line ~165, where `input_`
becomes the post-attention residual) and `hidden_states = self.norm_2(input_)`:

```python
with set_generator(generator):
    input_ = self._bias_dropout_add(hidden_states, bias, input_)
self._debug(input_, "mixer_residual", hidden_dims, kwargs)
kwargs[BlockKwargs.pre_mlp_residual] = input_      # NEW: only consumed by Gemma4MoEMLP
hidden_states = self.norm_2(input_)
self._debug(hidden_states, "norm_2", hidden_dims, kwargs)
hidden_states, bias = self.mlp(hidden_states, kwargs, losses, metrics)
kwargs.pop(BlockKwargs.pre_mlp_residual, None)     # NEW: drop reference so the tensor can be freed
```

The set+pop pair is per-block (overwritten every layer; cleared as soon as the MLP returns). This
is a no-op for every existing MLP type — they ignore the key. Must run unconditionally so the
kwarg never leaks into the next block.

---

## New layer type: `Gemma4MoEMLP`

### 3. Config — `fast_llm/layers/decoder/mlp/config.py`

Append after `MoEMLPConfig`:

```python
@config_class(dynamic_type={MLPBaseConfig: "gemma4_moe"})
class Gemma4MoEMLPConfig(MLPConfig):
    """
    Parallel dense + sparse MoE used by Gemma4 when `enable_moe_block=True`.
    The dense branch is the standard MLPConfig; the sparse branch operates on the
    raw pre-norm residual (read via BlockKwargs.pre_mlp_residual).
    """
    _abstract = False

    router: LinearConfig = Field(
        desc="Router projection (hidden_size -> num_experts).",
        hint=FieldHint.feature,
    )
    num_experts: int = Field(
        default=2,
        desc="Number of experts in the sparse branch.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 1),
    )
    experts_per_token: int = Field(
        default=1,
        desc="Top-k experts activated per token.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    moe_intermediate_size: int = Field(
        default=2048,
        desc="Per-expert intermediate size.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    post_feedforward_norm_1: RMSNormalizationConfig = Field(
        desc="Post-norm of the dense branch (HF: post_feedforward_layernorm_1).",
        hint=FieldHint.architecture,
    )
    pre_feedforward_norm_2: RMSNormalizationConfig = Field(
        desc="Pre-norm of the sparse branch on the raw residual (HF: pre_feedforward_layernorm_2).",
        hint=FieldHint.architecture,
    )
    post_feedforward_norm_2: RMSNormalizationConfig = Field(
        desc="Post-norm of the sparse branch (HF: post_feedforward_layernorm_2).",
        hint=FieldHint.architecture,
    )

    @property
    def layer_class(self) -> "type[Gemma4MoEMLP]":
        from fast_llm.layers.decoder.mlp.gemma4_moe import Gemma4MoEMLP
        return Gemma4MoEMLP
```

### 4. Module — `fast_llm/layers/decoder/mlp/gemma4_moe.py` (new file)

`Gemma4MoEMLP` extends `MLP` (so it inherits `layer_1` / `layer_2` for the dense branch), adds:
- `post_feedforward_norm_1`, `pre_feedforward_norm_2`, `post_feedforward_norm_2` (RMSNorms)
- `router_norm` — RMSNorm with `with_scale=False` (fixed-scale rms_norm before router projection)
- `router` (Linear hidden→num_experts; weight absorbs `hf_router.scale * hidden_size**-0.5` at import)
- `expert_layer_1` and `expert_layer_2` — stored in Fast-LLM's flat layout `(E*2I, H)` and `(E*I, H)`,
  reshaped from HF's `(E, 2I, H)` / `(E, H, I)` at import time

`_forward(input_, kwargs, ...)`:

```
dense_out  = mlp_autograd(input_, ..., self.layer_1.weight, self.layer_2.weight, ...)
dense_out  = self.post_feedforward_norm_1(dense_out)

residual   = kwargs[BlockKwargs.pre_mlp_residual]
flat       = residual.reshape(-1, hidden_size)

# Router: fixed-scale rms_norm → linear (with fused scale * H**-0.5) → softmax → topk-on-probs → sum-to-1
# IMPORTANT: this is *not* the same as Fast-LLM's standard MoE routing.
#   Fast-LLM MoE: top_logits, top_idx = topk(logits); scores = softmax(top_logits)
#   Gemma4:       probs = softmax(logits); top_w, top_idx = topk(probs); top_w /= top_w.sum()
# The two are not equivalent — softmax-then-topk uses denominator over ALL experts; topk-then-softmax
# uses denominator over only the selected experts. Must match HF exactly.
router_in  = self.router_norm(flat)
logits     = self.router(router_in)
probs      = softmax(logits, dim=-1, dtype=float32)
top_w, top_idx = probs.topk(experts_per_token, dim=-1)
top_w     /= top_w.sum(dim=-1, keepdim=True)

sparse_in  = self.pre_feedforward_norm_2(flat)
sparse_out = _gemma4_moe_dispatch(sparse_in, top_w, top_idx,
                                   self.expert_layer_1.weight, self.expert_layer_2.weight,
                                   gated=True, activation=ActivationType.silu)
sparse_out = sparse_out.reshape_as(residual)
sparse_out = self.post_feedforward_norm_2(sparse_out)

return dense_out + sparse_out, None
```

`_gemma4_moe_dispatch` is a thin helper that copies `MixtureOfExpertMLP._forward_dropless` /
`_forward_looped` (calling `mlp_autograd` / `mlp_autograd_looped` with `transposed_layer_2_weight=True`,
sparse_map built from `top_idx`). We **don't** call `MixtureOfExpertMLP` itself — its `_forward`
bundles routing (softmax/topk-without-renormalization, jitter, z_loss, aux_loss) inside the same
function, and Gemma4's router has different math (sum-to-1 normalization after topk, fixed-scale
pre-norm, no aux loss).

Conversion details (verified against `fast_llm/functional/triton/mlp.py` and Mixtral converter):

- `router.weight` ← `hf.router.proj.weight * (hf.router.scale * hidden_size**-0.5)[None, :]`
- `expert_layer_1.weight` (gate+up): HF `gate_up_proj` is `(E, 2I, H)` with each per-expert slice
  already in nn.Linear orientation `(out=2I, in=H)`. Reshape `(E, 2I, H) → (E*2I, H)` (no transpose).
  Fast-LLM `mlp_autograd_looped` calls `chunk_weight(weight_1, num_experts)` which chunks dim 0 into
  E pieces of `(2I, H)` — matches.
- `expert_layer_2.weight` (down): HF `down_proj` is `(E, H, I)` with each per-expert slice in
  nn.Linear orientation `(out=H, in=I)`. Fast-LLM `layer_2` uses `transposed_weight=True`, so each
  per-expert weight is stored as `(I, H)`. Per-expert: transpose `(H, I) → (I, H)`. Stack across
  experts: `(E*I, H)`. Mixtral's `MLPLayer2Converter` does this transpose (verified in
  `fast_llm/models/gpt/conversion/llama.py:MLPLayer2Converter`); Gemma4 can reuse it after slicing
  `down_proj[e]`.
- `per_expert_scale` ← fused into `expert_layer_2` rows. For each expert `e`, multiply that expert's
  block of `expert_layer_2` (rows `e*I .. (e+1)*I`, all H columns) by `hf.router.per_expert_scale[e]`.

> **Required test** (point 3 from review): write a unit test that constructs a tiny Gemma4 MoE block
> in HF, copies weights through this converter into `Gemma4MoEMLP`, and asserts both forward outputs
> AND input gradients match within `rms_close(1e-5)`. Layer-2 transpose direction is the most common
> place for silent bugs — the test must catch a swapped transpose.

**Roundtrip caveat**: fusing `per_expert_scale` into `expert_layer_2` is **inference-correct but not
state-dict-roundtrip-exact**. Exporting back to HF would produce `per_expert_scale = ones` and a
modified `down_proj`. Document this; refuse export until a separate parameter is wired in. Same
applies to `router.scale` fusion.

---

## Attention: per-layer-type via `PatternBlockSequenceConfig`

Build two `DecoderBlockConfig` instances:

| Block name | `head_dim` | `num_kv_heads` | Rotary | Sliding window | `attention_k_eq_v` applies? |
|------------|-----------|----------------|--------|----------------|-----------------------------|
| `sliding`  | 256       | `num_key_value_heads` | `DefaultRotaryConfig(theta=10_000)` | enabled | **No** under current HF behavior (has separate `v_proj`) |
| `full`     | 512       | `num_global_key_value_heads` | `ProportionalRotaryConfig(theta=1e6, partial_rotary_factor=0.25)` | disabled | **Yes** under current HF behavior (drops `v_proj`) |

> Note: the sliding-vs-full split for `attention_k_eq_v` follows HF's current expression
> `use_alternative_attention = config.attention_k_eq_v and not self.is_sliding`. This is HF's behavior
> in transformers 5.5.0, not an architectural law — verify in the reference file when bumping HF.

Pattern repeats `["sliding"] * 5 + ["full"]` for `num_blocks` total. The last layer is forced to
`full` (matches HF `__post_init__` warning).

Both blocks have `q_norm`, `k_norm` (with learnable scale, `RMSNormalizationConfig`) and
`value_norm=True` (no learnable scale; matches Gemma's `with_scale=False` v_norm).

The MoE flag is config-wide, so both block configs use either `MLPConfig` (dense-only) or
`Gemma4MoEMLPConfig` (dense+sparse) consistently.

### Handling `attention_k_eq_v=True` (target checkpoint sets this)

In HF, `use_alternative_attention = config.attention_k_eq_v and not self.is_sliding`. So:

- **Sliding layers**: unaffected — separate `k_proj`, `v_proj`, normal import.
- **Full-attention layers**: HF drops `v_proj` entirely, computes `value_states = k_proj(hidden_states)`
  (raw, before `k_norm`), then applies `v_norm`.

**Conversion strategy (inference-correct, no Fast-LLM changes needed)**: Fast-LLM has a fused
`key_value` projection — concatenate `[k_weight; v_weight]`. For full-attention layers under
`attention_k_eq_v=True`, set `v_weight = k_weight` (duplicate the storage). Fast-LLM applies
`k_norm` to the K half and `value_norm` (no scale) to the V half before attention, exactly matching
the HF order: `value_states = key_states_raw → v_norm → attention`.

Caveats:
- Doubles the parameter count of K/V on full-attention layers (memory cost).
- On export back to HF, the duplicated V weight will not equal `k_weight` after fine-tuning, so
  exporting an `attention_k_eq_v` checkpoint after Fast-LLM training would silently break the
  shared-projection assumption. Refuse export with `attention_k_eq_v=True` (or document the loss).
- A future optimization could add a Fast-LLM-side `attention_k_eq_v` config that wires V to K
  internally and avoids the duplication, but that's a separate change.

### Handling `use_bidirectional_attention="vision"` (target checkpoint sets this)

In HF text-only inference, `is_causal = config.use_bidirectional_attention != "all"`, so `"vision"`
maps to **causal** attention (no behavior change). The bidirectional behavior is applied externally
to vision tokens by `Gemma4Model` (see `modeling_gemma4.py` lines 2266 / 2537), not inside the
attention layer.

**Strategy**: accept `"vision"` and `None`; reject `"all"` (which Fast-LLM doesn't support without
a non-causal attention path). The text-only converter ignores the vision-mask machinery — the
Fast-LLM model is text-only anyway. Document that loading this checkpoint into a multimodal pipeline
will not produce HF-equivalent results for image tokens.

---

## Converter — `fast_llm/models/gpt/conversion/gemma4.py`

### `import_config(hf: Gemma4TextConfig) -> dict`

Reject features that have no inference-correct mapping; **accept** those needed by 26B A4B:

```python
# Hard rejects (no inference-correct path today):
if hf["hidden_size_per_layer_input"]:
    raise NotImplementedError("Gemma4 per-layer-input branch is not supported.")
if hf["num_kv_shared_layers"]:
    raise NotImplementedError("Gemma4 KV sharing across layers is not supported.")
if hf.get("use_double_wide_mlp"):
    raise NotImplementedError("Gemma4 use_double_wide_mlp is not supported.")
if hf.get("use_bidirectional_attention") == "all":
    raise NotImplementedError(
        "Gemma4 use_bidirectional_attention='all' requires non-causal attention; not supported."
    )

# Accept with handling:
# - attention_k_eq_v: handled per-layer-type at weight-conversion time (full-attention only).
# - use_bidirectional_attention=='vision' or None: text path is causal in HF; no-op for text-only Fast-LLM.

# Warn if non-default but functionally inert in text-only mode:
if hf.get("use_bidirectional_attention") == "vision":
    logger.warning("use_bidirectional_attention='vision' affects image tokens only; ignored in text-only converter.")
```

Map config:
- `vocab_size`, `num_hidden_layers`, `hidden_size`, `intermediate_size`, `rms_norm_eps` — direct
- `hidden_activation` → `ActivationType` (gelu_pytorch_tanh has a Fast-LLM equivalent)
- `layer_types` → `pattern` (sliding/full block selection)
- `head_dim` → sliding block, `global_head_dim` → full block
- `num_key_value_heads` → sliding block KV count
- `num_global_key_value_heads` (defaults to `num_key_value_heads` if `None`) → full block KV count
- `rope_parameters[layer_type]` → per-block rotary config
- `sliding_window` → sliding block attention config
- `final_logit_softcapping` → head config
- If `enable_moe_block`: `num_experts`, `top_k_experts`, `moe_intermediate_size` → both block MLPs are `Gemma4MoEMLPConfig`
- `tie_word_embeddings` → embedding config

The `Gemma4TextScaledWordEmbedding` multiplies by `sqrt(hidden_size)`. Fast-LLM embedding has an
equivalent scale factor — set it in `LanguageModelEmbeddingsConfig` at import.

### Weight converters

| HF key | Fast-LLM key | Notes |
|--------|--------------|-------|
| `model.embed_tokens.weight` | `embeddings.word_embeddings_weight` | tied to `head.output_weights` if enabled |
| `model.layers.{i}.input_layernorm.weight` | `decoder.{i}.norm_1.weight` | |
| `model.layers.{i}.self_attn.q_proj.weight` | `decoder.{i}.mixer.query.weight` | per-block: head_dim differs |
| `model.layers.{i}.self_attn.k_proj.weight` + `v_proj.weight` | `decoder.{i}.mixer.key_value.weight` | concat via `KeyValueWeightConverter` (sliding layers) |
| `model.layers.{i}.self_attn.k_proj.weight` (no v_proj) | `decoder.{i}.mixer.key_value.weight` | full-attention + `attention_k_eq_v=True`: duplicate k into v half |
| `model.layers.{i}.self_attn.o_proj.weight` | `decoder.{i}.mixer.dense.weight` | |
| `model.layers.{i}.self_attn.q_norm.weight` | `decoder.{i}.mixer.q_norm.weight` | module attribute is `q_norm` (not `query_norm`) |
| `model.layers.{i}.self_attn.k_norm.weight` | `decoder.{i}.mixer.k_norm.weight` | module attribute is `k_norm` (not `key_norm`) |
| (no HF weight) | `value_norm` is `with_scale=False` | enabled via `value_norm=True` in attention config; no weight to load |
| `model.layers.{i}.post_attention_layernorm.weight` | `decoder.{i}.post_mixer_normalization.weight` | |
| `model.layers.{i}.pre_feedforward_layernorm.weight` | `decoder.{i}.norm_2.weight` | |
| `model.layers.{i}.mlp.gate_proj.weight` + `up_proj.weight` | `decoder.{i}.mlp.layer_1.weight` | concat (gate, up) via `MLPLayer1Converter` |
| `model.layers.{i}.mlp.down_proj.weight` | `decoder.{i}.mlp.layer_2.weight` | transpose via `MLPLayer2Converter` |
| `model.layers.{i}.post_feedforward_layernorm.weight` | `decoder.{i}.post_mlp_normalization.weight` | |
| **MoE-only** | | |
| `model.layers.{i}.post_feedforward_layernorm_1.weight` | `decoder.{i}.mlp.post_feedforward_norm_1.weight` | |
| `model.layers.{i}.pre_feedforward_layernorm_2.weight` | `decoder.{i}.mlp.pre_feedforward_norm_2.weight` | |
| `model.layers.{i}.post_feedforward_layernorm_2.weight` | `decoder.{i}.mlp.post_feedforward_norm_2.weight` | |
| `model.layers.{i}.router.proj.weight` (+ `router.scale`) | `decoder.{i}.mlp.router.weight` | fused: `proj * scale[None,:] * H**-0.5` |
| `model.layers.{i}.router.per_expert_scale` | merged into expert layer_2 | `expert_layer_2[e] *= per_expert_scale[e]` |
| `model.layers.{i}.experts.gate_up_proj` | `decoder.{i}.mlp.expert_layer_1.weight` | reshape to Fast-LLM flat layout |
| `model.layers.{i}.experts.down_proj` | `decoder.{i}.mlp.expert_layer_2.weight` | reshape + transpose |
| `model.norm.weight` | `head.final_norm.weight` | |
| `lm_head.weight` | `head.output_weights` | dropped on import if `tie_word_embeddings` |

The `layer_scalar` buffer (per-layer learnable scalar applied at the end) is currently always 1.0 in
the released checkpoint. Treat as `IgnoreImportWeightConverter` for now and assert it equals 1.0
during import; raise if any value differs.

### Registration

`fast_llm/models/gpt/conversion/config.py`:
```python
class Gemma4CheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: ClassVar[str] = "gemma4"
```

`fast_llm/models/gpt/conversion/auto.py`:
```python
from fast_llm.models.gpt.conversion.gemma4 import Gemma4HuggingfaceCheckpointHandler
handler_map["gemma4"] = Gemma4HuggingfaceCheckpointHandler
```

---

## File summary (revised)

| File | Action | Est. lines |
|------|--------|-----------|
| `fast_llm/layers/block/config.py` | +1 line: `pre_mlp_residual` kwarg key | +1 |
| `fast_llm/layers/decoder/block.py` | +1 line: `kwargs[BlockKwargs.pre_mlp_residual] = input_` | +1 |
| `fast_llm/layers/decoder/mlp/config.py` | new `Gemma4MoEMLPConfig` (corrected: `Field`, `dynamic_type`, `layer_class`, `RMSNormalizationConfig`) | +60 |
| `fast_llm/layers/decoder/mlp/gemma4_moe.py` | new `Gemma4MoEMLP` (returns `(out, None)`; copies `_forward_dropless` / `_forward_looped` as helpers — no `MixtureOfExpertMLP` subclassing) | +180 |
| `fast_llm/models/gpt/conversion/gemma4.py` | new converter (uses `PatternBlockSequenceConfig`; explicit rejects for unsupported features) | +320 |
| `fast_llm/models/gpt/conversion/config.py` | `Gemma4CheckpointFormat` | +5 |
| `fast_llm/models/gpt/conversion/auto.py` | register handler | +3 |
| **Total** | | **~570** |

---

## Explicitly out of scope (raise on import)

- `hidden_size_per_layer_input != 0` / per-layer-input branch
- `num_kv_shared_layers != 0` (cross-layer KV sharing)
- `use_double_wide_mlp=True` (fused gate+up MLP variant)
- `use_bidirectional_attention == "all"` (requires non-causal attention)
- `layer_scalar != 1.0` (verify on import; ignore otherwise)
- Vision and audio sub-configs (text-only converter)
- **Export** of an `attention_k_eq_v=True` checkpoint after Fast-LLM training (V duplicate has
  diverged from K; refuse to write `attention_k_eq_v=True` export)
- **Export** of MoE checkpoint after fine-tuning (router scale and per-expert scale are fused into
  weights at import; export would be lossy — refuse until separate-parameter path is added)

## Accepted with caveats (used by 26B A4B)

- `attention_k_eq_v=True`: full-attention layers duplicate `k_proj` weight into V slot at import.
- `use_bidirectional_attention="vision"`: text path is causal (HF agrees); vision-token bidirectional
  behavior is not modeled in text-only Fast-LLM converter.
- `enable_moe_block=True` with `top_k_experts`, `num_experts`, `moe_intermediate_size`.

For the released `e2b-it` / `26b-a4b` checkpoints we still need to read the actual config files and
add tests that exercise the rejection logic (so we fail loudly, not silently produce a degraded
model).

---

## Required tests (must land with the implementation, not as follow-up)

### 1. `tests/layers/test_gemma4_moe.py` — HF reference parity (MLP subpath only)

Scope: test only `Gemma4MoEMLP` in isolation — **not** a full `Gemma4TextDecoderLayer`. This
avoids needing to wire attention masks, position embeddings, and the pre/post norms that live
outside the MLP.

Setup:
- Build a tiny `Gemma4TextConfig` with `enable_moe_block=True`.
- Instantiate HF's `Gemma4TextExperts`, `Gemma4TextRouter`, and the three inner RMSNorms directly
  from the config.
- Port their weights into a `Gemma4MoEMLP` using the converter's import logic.
- Also port the dense MLP (`Gemma4TextMLP`) and replicate the full HF feedforward computation in a
  reference function: `dense(norm_2(hidden)) → post_norm_1 → sparse(pre_norm_2(residual)) → sum`.
- Call `Gemma4MoEMLP._forward(norm_2_out, kwargs)` with `kwargs[pre_mlp_residual] = residual`.

Assertions:
- `forward` outputs match within `rms_close(1e-5)` for fp32 inputs.
- Input `.grad` matches within `rms_close(1e-4)` after a backward on a sum-loss.
- Compare both at `experts_per_token=1` and `experts_per_token=2`.
- Include a fixed-seed input producing non-trivial top-2 probabilities so the sum-to-1
  renormalization does real work; test must FAIL if routing uses the standard Fast-LLM
  `topk(logits) → softmax(top_logits)` form instead.

### 2. `tests/models/test_gemma4_converter.py` — `attention_k_eq_v` weight wiring

Two assertions per layer-type:

- **Sliding layer** (HF `is_sliding=True`): assert that the imported `key_value.weight` equals
  `cat(hf.k_proj.weight, hf.v_proj.weight, dim=0)`. End-to-end attention output must match HF.
- **Full-attention layer** (HF `is_sliding=False`, `attention_k_eq_v=True`, `v_proj is None`):
  assert imported `key_value.weight = cat(hf.k_proj.weight, hf.k_proj.weight, dim=0)`. End-to-end
  attention output must match HF — this verifies the V branch goes through `value_norm` (no scale)
  on a copy of the raw K projection, not on the post-`k_norm` output.

A second forward-only test should construct a 6-layer model with the canonical `5:1` pattern,
both layer types sharing weights, and verify per-layer outputs match HF for an input where the
sliding window is exercised (sequence longer than `sliding_window`).

### 3. `tests/layers/test_gemma4_moe.py::test_gemma4_router_math` (must-have)

Standalone test isolating just the router:

```python
# Build a 4-expert, top-2 router; feed a (T, H) tensor with hand-picked logits so that:
# - HF top-k weights renormalize to sum 1 per token
# - The naive Fast-LLM topk-then-softmax form produces different weights
# Assert HF reference values; test should fail if the implementation drifts.
```

## Open questions

1. `per_expert_scale` and `router.scale` fusion vs separate parameters — fusing is simpler and
   inference-correct, but **export is lossy**. Recommendation: fuse for the inference-only converter
   v1; add separate parameter support later if fine-tuning + re-export is needed.
2. Expert weight layout — store flat `(E*2I, H)` / `(E*I, H)` at import and copy
   `_forward_dropless` / `_forward_looped` as helpers (no `MixtureOfExpertMLP` subclassing).
3. `attention_k_eq_v` import strategy — duplicate K weight into V slot (current plan, simple) vs.
   add a Fast-LLM-side `key_value_shared` config and fuse internally (smaller memory). Recommendation:
   duplicate for v1; optimize later.
4. Whether to also support `enable_moe_block=False` checkpoints in the same converter (just skip the
   MoE-only weight lines and use plain `MLPConfig`). Recommendation: yes — branch on the config flag
   at `import_config`.
