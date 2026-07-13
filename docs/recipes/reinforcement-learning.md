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

The **operative config is a single committed file on the orchestrator side** — one hydra
config that pins every hyperparameter for the run (both the orchestrator/DeepSpeed settings
and the Fast-LLM `fast_llm:` subtree) and is what the launcher composes; the launch command
itself carries no hyperparameters, only environment (paths, W&B identity). That file is the
source of truth for a run.

Two illustrative excerpts accompany this page as a readable reference for the Fast-LLM half
of that config — they are **not** the operative source and **not** standalone-runnable (the
model architecture is imported from the Qwen2.5 checkpoint and the data comes from the
stream); treat them as documentation of the knobs, and edit the orchestrator hydra config to
actually change a run:

- [`examples/qwen_gspo_0.5b.yaml`](https://github.com/ServiceNow/Fast-LLM/blob/main/examples/qwen_gspo_0.5b.yaml)
- [`examples/qwen_gspo_7b.yaml`](https://github.com/ServiceNow/Fast-LLM/blob/main/examples/qwen_gspo_7b.yaml)

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
- **`metrics`** — opts into diagnostic metrics; shared by both GSPO and GRPO. `none`
  (default) logs nothing extra; `basic` logs importance-ratio, KL and advantage statistics
  plus the rollout reward and staleness; `with_entropy` also logs the policy entropy. The
  extra statistics need a second full-vocab softmax, so `basic`/`with_entropy` require
  `pipeline_parallel == 1` and add a compute cost that is skipped entirely at `none`. This
  is the switch to turn on when measuring lag and instability — see
  the **Measuring lag and instability** section.

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
| `zero_stage` | `1` | `2` |
| `mlp.recompute_level` | — | `full` |
| Trainer sharding | 1 replica (FSDP size 1) | 4 replicas (FSDP size 4) |

At FSDP size 1 (the 0.5B trainer) ZeRO shards across nothing, so the stage is a no-op —
`1` is simply the minimal setting. On the 4-GPU 7B trainer, ZeRO-2 shards optimizer state
and gradients (enough) and avoids ZeRO-3's per-step parameter all-gather; go to `3` only if
it OOMs (the fp32 lm-head logits are the large transient).

Each size was run in three precision arms — **bf16** (default), **fp16-matched**, and
**fp32-matched** — alongside a **DeepSpeed** parity baseline (see
the **DeepSpeed parity baseline** section).

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

For A/B comparison against a DeepSpeed trainer, match these on the DeepSpeed side:

| Fast-LLM | DeepSpeed equivalent |
| --- | --- |
| `optimizer.learning_rate.base: 5e-7` | `learning_rate: 5e-7` |
| `learning_rate.warmup_iterations: 0` | `num_warmup_steps: 0` |
| `learning_rate.decay_style: constant` | `lr_scheduler_type: constant` |
| `schedule.docs_per_step: 256` | `gradient_accumulation_passes: 256` |
| `optimizer.beta_1/beta_2` | Adam `beta1/beta2` (same values) |
| `fp32_lm_head: true` | fp32 LM head (matched) |
| `multi_stage.zero_stage: 1 / 2` | `deepspeed_config: deepspeed_stage{1,2}_bf16` |

The orchestrator's DeepSpeed trainer picks its ZeRO stage from its own DeepSpeed config
(a `deepspeed_config` selecting a stage JSON), independently of the Fast-LLM `zero_stage`.
The ZeRO stage is a memory/sharding choice and is mathematically equivalent across stages,
so it does **not** affect the reward comparison and the two trainers need not match — pick
the lowest stage that fits (stage 1 for a single-GPU trainer, stage 2 for the 4-GPU 7B
trainer). The samplers, temperature, `epsilon`, seed, and sequence shape are identical
across both trainers.

## 📊 Measuring lag and instability

Set `metrics: basic` on the policy-gradient loss to log the diagnostics needed to quantify
rollout staleness and training instability (`with_entropy` adds the policy entropy on top).
The statistics are gated behind `metrics != none`, so `none` logs none of them.

Enabled by `metrics: basic` (each prefixed with the loss name, e.g. `gspo_`), logged as
mean plus max/min where applicable:

| Metric | Meaning |
| --- | --- |
| `staleness` | `documents_seen − model_version`: documents trained since each token's generating policy was synced — the rollout **lag**. `max_staleness` is the worst-case lag. |
| `train_samples_reward` | Mean rollout reward. Averaged over the sample-filtered batch it is biased, so treat it as a diagnostic, not a policy-performance metric. |
| `clipped_ratio_fraction` | Fraction of segments (GSPO) / tokens (GRPO) whose importance ratio was clipped — the primary instability indicator. |
| `ratio_new_old`, `ratio_new_old_sum`, `ratio_new_old_squared_sum` | Importance-ratio mean and the sums to derive its variance. |
| `kl_new_old` | KL(new ‖ old) between the trainer and sampler policies. |
| `advantage`, `max_advantage`, `min_advantage` | Advantage statistics. |
| `num_tokens` | Labeled-token count, for per-token averages. |
| `entropy` | Policy entropy (only with `metrics: with_entropy`). |

The streaming trainer additionally logs, unconditionally, `weight_sync_time_ms` (wall-clock
cost of broadcasting weights to the samplers each step) and `documents_seen` (cumulative
document count, also selectable as a W&B x-axis).

GSPO's ratio/clip statistics are segment-level and GRPO's are token-level, so the suffixes
match where the meaning lines up but the units differ.

## 🧷 Configuration gotchas

- **`micro_batch_size` is in tokens and must be ≥ the packed sequence length.** One
  sequence per micro-batch, so it tracks the sequence length (10000 here) and cannot be
  set smaller. Lowering trainer memory means lowering sequence length and
  `micro_batch_size` together.
- **Layering over an orchestrator-generated config.** If the orchestrator already emits a
  Fast-LLM config and you override via the command line, use add-or-override (`++`) for
  keys it already sets (e.g. `multi_stage.zero_stage`) and add (`+`) only for keys it does
  not.
- **`metrics` requires `pipeline_parallel == 1`.** The diagnostic metrics compute a second
  full-vocab softmax, which pipeline parallelism would split; setting `basic`/`with_entropy`
  under `pipeline_parallel > 1` is rejected. The reference runs use `pipeline_parallel == 1`.
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
- **Diagnostic metrics for GSPO and GRPO.** The `metrics` field is shared by both losses,
  so GSPO runs can log importance-ratio, KL, advantage, clip-fraction and (optionally)
  entropy statistics — see the **Measuring lag and instability** section.
- **Document / reward / model-version metrics.** The trainer tracks `documents_seen`
  (checkpointed and offered as a W&B x-axis) and carries the reward and a per-token model
  version through the batch, from which the reward and rollout staleness are logged.

## 📌 Open items

- **Current focus: newer sampler + measure lag.** The 7B runs show reward spikes traced to
  rollout staleness and an abort storm when in-flight generations are cancelled on a weight
  sync — *not* to numerical noise (fp16 and bf16 spike at the same rate). The immediate step
  is to re-run on a sampler build that keeps in-flight generations across a weight update
  (instead of aborting them), with `metrics: basic` enabled, and quantify the remaining
  staleness and clip fraction. See
  the **Measuring lag and instability** section.
- **Later, once the metrics are in hand.** Candidate mitigations — to be judged against the
  measured staleness rather than adopted blind — bound the rollout lag, raise the
  weight-update interval, and double-buffer weights. Deferred until the lag measurements exist.
