# Batch-size scaling experiment (Qwen2.5-0.5B)

Tests whether small-batch training matches large-batch when Adam hyperparameters are scaled
correctly (Marek et al., arXiv:2507.07101): only β2 must scale to hold the token half-life
fixed; β1 stays at 0.9; the learning rate is ~batch-insensitive.

Loss is compared **vs tokens seen**. All arms share one model init (the warmup checkpoint) and
one fresh, disjoint token stream (the `experiment` split), so the curves form a paired comparison.

## Arms (b=16, B=512, ratio 32; lengths in sequences of 2048 tokens)

| Arm | batch | lr | β1 | β2 | role |
|---|---|---|---|---|---|
| A | 512 | 1e-4 | 0.9 | 0.95 | large-batch reference |
| B | 16 | 1e-4 | 0.9 | 0.998398 (= 0.95^(1/32)) | paper scaling — only β2 moves |
| C | 16 | 3.125e-6 (= 1e-4/32) | 0.996875 | 0.9984375 | conservative — linear lr + linear (1−β) |
| D | 16 | 1e-4 | 0.9 | 0.95 | naive unscaled (strawman) |

lr held constant (no decay) at 1e-4 — the Qwen2.5-7B peak is 3e-4 cosine-to-10%; we use a lower
constant value since constant-lr has no decay phase to lower the effective rate.

Predictions: B overlays A; D degrades (wrong steady-state second-moment averaging); C sits above
B if scaling lr down 32× undertrains. `arm_base.yaml` defaults to D.

## Sequence

```bash
# 0. Tokenize FineWeb-Edu -> 3 disjoint splits
fast-llm prepare gpt_memmap --config prepare.yaml

# 1. Warmup (4 GPUs, ~1h). Kill at ~1h; note the latest checkpoint iteration <ITER>.
torchrun --nproc_per_node=4 -m fast_llm train gpt --config warmup.yaml

# 2. Set pretrained.path in arm_base.yaml to experiments/batch_size/warmup/checkpoint/<ITER>,
#    then launch the four arms in parallel, one GPU each:
CUDA_VISIBLE_DEVICES=0 fast-llm train gpt --config arm_base.yaml \
  run.experiment_dir=experiments/batch_size/arm_A \
  schedule.depth_first_micro_batches=512 training.train_iters=40000 \
  training.checkpoint.interval=1000 training.evaluators.validation.interval=190 &

CUDA_VISIBLE_DEVICES=1 fast-llm train gpt --config arm_base.yaml \
  run.experiment_dir=experiments/batch_size/arm_B \
  optimizer.beta_2=0.998398 &

CUDA_VISIBLE_DEVICES=2 fast-llm train gpt --config arm_base.yaml \
  run.experiment_dir=experiments/batch_size/arm_C \
  optimizer.beta_2=0.9984375 optimizer.beta_1=0.996875 optimizer.learning_rate.base=3.125e-6 &

CUDA_VISIBLE_DEVICES=3 fast-llm train gpt --config arm_base.yaml \
  run.experiment_dir=experiments/batch_size/arm_D &
```

`num_samples` is identical across arms (512×40000 = 16×1280000 = 20,480,000 ≈ 42B tokens, within the
~94B `experiment` split), so the shuffled token stream is identical — only the batching differs. The
42B is a cap; monitor the curves and stop when the comparison is conclusive. Compare loss at matched
tokens (step × micro_batch_size × depth_first), as a paired difference vs A plus held-out validation loss.

Logging goes to W&B `servicenow-team/batch_size_experiments` (group `arms`; warmup in group `warmup`);
each arm is a separate run named after its experiment_dir.
