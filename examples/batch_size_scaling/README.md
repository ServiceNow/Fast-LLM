# Batch-size scaling experiment (Qwen2.5-0.5B)

Does small-batch Adam training **reproduce** large-batch training when the hyperparameters are scaled
by the square-root (SDE) rule, and how does that compare to the "keep lr, scale β2" paper rule? See
[`ANALYSIS.md`](ANALYSIS.md) for the theory, the predictions, and the results.

Loss is compared **vs tokens seen** (`training/consumed_tokens`) — optimizer steps are not comparable
across batch sizes. All arms share one model init (the warmup checkpoint) and the same shuffled token
stream (the `experiment` split), so the curves form a paired comparison.

## Reproduce

```bash
# 0. Tokenize FineWeb-Edu into disjoint warmup / experiment / validation splits.
fast-llm prepare gpt_memmap --config prepare.yaml

# 1. Throwaway warmup from scratch (~1B tokens); note the final checkpoint iteration.
torchrun --nproc_per_node=4 -m fast_llm.cli train gpt --config warmup.yaml

# 2. Branch each arm from the warmup checkpoint (weights only, cold optimizer) via `pretrained`,
#    one run per arm. arm_base.yaml holds the shared settings; arms differ only in
#    schedule.depth_first_micro_batches (batch), optimizer.learning_rate.base, and the betas.
fast-llm train gpt --config arm_base.yaml \
  run.experiment_dir=experiments/batch_size/<arm> \
  schedule.depth_first_micro_batches=<n> \
  optimizer.learning_rate.base=<lr> optimizer.beta_1=<b1> optimizer.beta_2=<b2>
```

Set `pretrained.path` to the warmup checkpoint and the W&B `entity_name` / `project_name` in the
configs to your own. Batch (tokens) = `depth_first_micro_batches × micro_batch_size × data_parallel`;
`b` below is in sequences of 2048.

## Arms

Two √-rule pairs anchor the comparison — scale **down** from the large-batch baseline (A → H) and **up**
from the small-batch keep-lr arm (B → J). With batch ratio `r`:

| arm | batch | lr | β1 | β2 | what it isolates |
|---|---|---|---|---|---|
| A | large | 1e-4 | 0.9 | 0.95 | large-batch baseline |
| H | small | 1e-4/√r | 0.9^(1/r) | 0.95^(1/r) | **full √/SDE rule** (= A scaled down) |
| I | small | 1e-4/√r | 0.9 | 0.95^(1/r) | √-lr + β2 only (drops β1) |
| B | small | 1e-4 | 0.9 | 0.95^(1/r) | keep-lr + β2 (paper rule) |
| M | small | 1e-4 | 0.9^(1/r) | 0.95^(1/r) | keep-lr + both β |
| J | large | 1e-4·√r | 0.9 | 0.95 | large-batch √-up image of B |
| K | large | (lr sweep) | 0.9 | 0.95 | large-batch lr sweep |
| L | large | 1e-4 | 0.9 | 0.95 | fp16 twin of A (precision) |

All arms: constant lr (no decay), `weight_decay=0`, `epsilon=1e-8`, gradient-norm clip 5.0, shared
seed. `num_samples` (= `depth_first_micro_batches × train_iters`) is matched across arms so the shuffled
stream is identical — only the batching differs.

Two rounds were run (details and results in [`ANALYSIS.md`](ANALYSIS.md)): round 1 at `r=32` early in
training (**signal-dominated** — the wrong regime, the rule's knobs wash out), and round 2 at `r=16`
deep in training (**noise-dominated** — where the rule holds and the √-pairs overlay).

> **Gotcha — don't reuse the data seed when branching.** The document shuffle is seeded independently
> of batch size / `num_samples`, so a child run with the same seed *replays the parent's already-seen
> data* until it reaches the parent's token horizon (a sharp, persistent loss step). Use a different
> seed, or skip the parent's consumed prefix. See the methodology note in `ANALYSIS.md`.
