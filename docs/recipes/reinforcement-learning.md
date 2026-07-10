---
title: Reinforcement Learning with GSPO / GRPO
---

Fast-LLM can act as the **trainer** in an asynchronous reinforcement-learning loop. An
external RL orchestrator generates rollouts with [vLLM](https://github.com/vllm-project/vllm)
samplers, scores them into per-token advantages, and streams the packed rollouts to the
Fast-LLM trainer; after each optimizer step the updated weights are broadcast back to the
samplers over NCCL. Fast-LLM itself only sees packed sequences that carry, per document,
the **advantage**, the **sampling log-probabilities** (the behaviour policy `π_old`), and
the **document lengths**. The policy-gradient objective (GSPO or GRPO) is computed in the
language-model head.

This recipe documents the training-side configuration used for a set of reference
Qwen2.5 experiments, and the reasoning behind each setting. The orchestration layer, the
rollout data pipeline, and the reward model are out of scope — this page is about how to
configure the Fast-LLM trainer.

Two ready-to-adapt override fragments accompany this page:

- [`examples/qwen_gspo_0.5b.yaml`](https://github.com/ServiceNow/Fast-LLM/blob/main/examples/qwen_gspo_0.5b.yaml)
- [`examples/qwen_gspo_7b.yaml`](https://github.com/ServiceNow/Fast-LLM/blob/main/examples/qwen_gspo_7b.yaml)

They contain only the RL-relevant overrides; the base model config (architecture, vocab,
tokenizer, RoPE) is imported from the Qwen2.5 checkpoint by the orchestrator, and the
data stream is supplied at run time.

## 🎯 The policy-gradient loss

The RL objective is one entry in the head's `losses` dictionary. The dictionary key is a
free label; the `type` field selects the loss:

```yaml
model:
  base_model:
    head:
      fp32_lm_head: true
      losses:
        gspo:                       # free label; use `grpo` for GRPO
          type: gspo                # `gspo` or `grpo`
          epsilon_low: 0.003        # lower clip on the log-prob ratio
          epsilon_high: 0.004       # upper clip on the log-prob ratio
          logits_scale_factor: 1.4285714
```

- **`type`** — `gspo` (sequence-level importance ratio, geometric-mean per segment) or
  `grpo` (token-level ratio). Both derive from `LanguageModelPolicyGradientLossConfig`
  and share the fields below.
- **`epsilon_low` / `epsilon_high`** — the asymmetric clip on the importance ratio. GSPO
  operates on a per-segment geometric-mean ratio, so its useful epsilon range is far
  tighter than GRPO's token-level clip; the reference runs use `0.003 / 0.004`.
- **`logits_scale_factor`** — set to `1 / sampling_temperature` so the trainer's
  log-probabilities are computed at the same temperature the sampler used. With sampling
  temperature `0.7` this is `1.4285714`. Leaving it at the default `1.0` while sampling at
  `T < 1` inflates the importance ratio and destabilizes training.
- **`metrics`** — *(GRPO only.)* Opts into diagnostic metrics (clip fraction, entropy,
  ratio statistics). This field lives on the GRPO config and is **rejected by a `gspo`
  loss**; do not set it on a GSPO run. See [Open items](#-open-items).

### `fp32_lm_head`

`fp32_lm_head: true` keeps the language-model head in fp32 regardless of the body compute
dtype. The samplers compute their logits through an fp32 path, and matching the trainer
head to it removes a source of cross-engine numerical drift. This is independent of the
body precision — keep it on for bf16, fp16, and fp32 runs alike.

## ⚙️ Reference recipe

The reference experiments train Qwen2.5-0.5B and Qwen2.5-7B on a math-reasoning task with
GSPO. The shared recipe:

| Setting | Value | Notes |
| --- | --- | --- |
| Loss | GSPO, `epsilon_low=0.003`, `epsilon_high=0.004` | |
| `logits_scale_factor` | `1.4285714` | `= 1 / 0.7` |
| `fp32_lm_head` | `true` | |
| `docs_per_step` | `256` | rollouts accumulated per optimizer step |
| `depth_first_micro_batches` | `48` | |
| `micro_batch_size` | `10000` tokens | tracks the packed sequence length |
| Learning rate | `5e-7`, constant, `0` warmup | |
| `beta_1` / `beta_2` | `0.974004` / `0.999750` | sqrt-rule, `m=4` |
| `gradient_norm_clipping` | `76.8` | `= 0.3 × docs_per_step` |
| Seed | `43` | |
| Sampling temperature | `0.7` | |

Per-size differences:

| | Qwen2.5-0.5B | Qwen2.5-7B |
| --- | --- | --- |
| Head | tied | untied |
| `zero_stage` | `2` | `3` |
| `mlp.recompute_level` | — | `full` |
| Trainer sharding | 1 replica (FSDP size 1) | 4 replicas (FSDP size 4) |

Each size was run in three precision arms — **bf16** (default), **fp16-matched**, and
**fp32-matched** — alongside a **DeepSpeed** parity baseline (see
[DeepSpeed parity](#-deepspeed-parity-baseline)).

### Derived quantities

- **`gradient_norm_clipping = 0.3 × docs_per_step`.** The per-step gradient is normalized
  by the document count, so the raw gradient norm scales with `docs_per_step`. To hold the
  *effective* clip fixed at `0.3`, the threshold must scale the same way: `0.3 × 256 = 76.8`.
  If you change `docs_per_step`, rescale the clip.
- **`docs_per_step` should equal the number of rollouts per step.** The loss divides by the
  document count seen across the accumulated micro-batches; setting `docs_per_step` to the
  rollout count makes that divisor equal to the batch size, matching the reference-engine
  normalization.
- **Adam betas follow the sqrt-rule for an effective batch multiplier `m`:**
  `beta = beta_default ** (1/m)`. With `m = 4`, `0.9 → 0.974004` and `0.999 → 0.999750`.

## 🔬 Precision matching

The single most important correctness constraint: **the trainer compute dtype must match
the sampler dtype.** The trainer recomputes `π` on the rollouts to form the importance
ratio `exp(new_logprob − old_logprob)`; `old_logprob` comes from the sampler. If the two
engines run at different precisions, the ratio is systematically biased even at step 0,
which corrupts the clip decisions and collapses reward.

- **bf16 (default).** Trainer body bf16, sampler bf16, head fp32 on both. This is the
  baseline and needs no precision overrides.
- **fp16-matched / fp32-matched.** Set `model.distributed.compute_dtype` to `float16` /
  `float32` **and** set the sampler dtype to the same value. Do not set only one — a naive
  `compute_dtype=float16` with a bf16 sampler is the mismatch failure above (reward drops
  to noise and throughput falls sharply from the diverging clip mask).

Precision *level* is not the driver of the trainer↔sampler gap; precision *mismatch* is.
In the reference runs the matched fp16 and fp32 arms track (and slightly beat) bf16, which
is consistent with bf16's 8-bit mantissa sitting near the noise floor of the very tight
GSPO epsilon.

## 🧮 DeepSpeed parity baseline

For A/B comparison against a DeepSpeed ZeRO-3 trainer, match these on the DeepSpeed side:

| Fast-LLM | DeepSpeed equivalent |
| --- | --- |
| `optimizer.learning_rate.base: 5e-7` | `learning_rate: 5e-7` |
| `learning_rate.warmup_iterations: 0` | `num_warmup_steps: 0` |
| `learning_rate.decay_style: constant` | `lr_scheduler_type: constant` |
| `schedule.docs_per_step: 256` | `gradient_accumulation_passes: 256` |
| `optimizer.beta_1/beta_2` | Adam `beta1/beta2` (same values) |
| `fp32_lm_head: true` | fp32 LM head (matched) |

DeepSpeed runs ZeRO-3 bf16 with gradient checkpointing. The samplers, temperature,
`epsilon`, seed, and sequence shape are identical across both trainers.

## 🧷 Configuration gotchas

- **`micro_batch_size` is in tokens and must be ≥ the packed sequence length.** One
  sequence per micro-batch, so it tracks the sequence length (10000 here) and cannot be
  set smaller. Lowering trainer memory means lowering sequence length and
  `micro_batch_size` together.
- **Layering over an orchestrator-generated config.** If the orchestrator already emits a
  Fast-LLM config and you override via the command line, use add-or-override (`++`) for
  keys it already sets (e.g. `multi_stage.zero_stage`) and add (`+`) only for keys it does
  not.
- **GSPO rejects `metrics`.** The `metrics` field exists on the GRPO config only; a `gspo`
  loss raises a validation error if it is set. Leave it unset for GSPO.
- **`normalize_by_documents` no longer exists.** Document normalization is applied
  unconditionally; there is no config knob for it.

## 🆕 Relevant trainer behavior

- **Gradient-accumulation fix (single-replica trainers).** A prior bug suppressed the
  decoder gradient under multi-micro-batch accumulation when the trainer used a single
  FSDP shard with shared gradient buffers. This is fixed. It only affected FSDP-size-1
  trainers (the 0.5B reference run); sharded trainers (the 7B run, FSDP size 4) were never
  affected.
- **`docs_per_step` accumulation.** The scheduler accumulates micro-batches until
  `docs_per_step` documents are seen, then takes one optimizer step, so the loss divisor is
  the full step's document count rather than a single micro-batch's.
- **Document / reward / model-version metrics.** The trainer tracks `documents_seen`
  (checkpointed and offered as a W&B x-axis) and carries the reward and a per-token model
  version through the batch, logging reward and version statistics.

## 📌 Open items

Known limitations and proposed follow-ups for the reference experiments:

- **Diagnostic metrics for GSPO.** The `metrics` field is GRPO-only today; exposing it on
  the shared policy-gradient base would let GSPO runs log clip fraction, entropy, and ratio
  statistics. This is a pending code change, not a config option.
- **7B rollout-staleness instability.** The 7B runs show reward spikes traced to rollout
  staleness and an abort storm when in-flight generations are cancelled on weight sync —
  *not* to numerical noise (fp16 and bf16 spike at the same rate). Proposed mitigations
  (not yet adopted): bound rollout lag, raise the weight-update interval, avoid aborting
  in-flight generations on sync, and double-buffer weights.
