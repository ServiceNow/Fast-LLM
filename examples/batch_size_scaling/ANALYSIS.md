# Batch-size scaling for Adam — analysis

Companion to `README.md` (which gives the run matrix). This note records *why* the experiment
is designed the way it is: what scaling of Adam's hyperparameters with batch size is correct, why
the naive intuitions fail, and what we can claim with a guarantee versus only empirically.

## Motivation

The folklore is that large batches are necessary for good training. The gradient-noise-scale theory
(McCandlish et al., [arXiv:1812.06162](https://arxiv.org/abs/1812.06162)) already says otherwise:
there is a *critical batch size* below which adding batch buys almost nothing. Large batch is also
actively painful in practice — in RL it forces stale rollouts and heavy gradient accumulation.

So: can small-batch training match large-batch if Adam's hyperparameters are scaled correctly — and
is there a scaling rule we can trust *before* tuning, in regimes we can't cheaply sweep (late
training, RL)? We want an **equivalence** rule with a generalization guarantee, not just a setting
that happens to be optimal in one regime.

## Setup

Adam's per-coordinate update is **normalized**:

```
θ ← θ − lr · m̂ / (√v̂ + ε)
m = β1·m + (1−β1)·g       (first moment  → smoothed gradient, the "signal")
v = β2·v + (1−β2)·g²      (second moment → recent mean-square, per-coordinate scale)
```

(`m̂`, `v̂` are bias-corrected.) Write the minibatch gradient as `g = ⟨g⟩ + n`, signal plus
zero-mean noise, with batch-noise variance `⟨n²⟩ = σ²/b` for batch size `b`.

The key fact is that `m̂/√v̂ ≈ O(1)` per coordinate (the gradient magnitude is divided out), so the
**step size is set by `lr` itself**, not by `lr × gradient`.

## Why the SGD linear-scaling rule does **not** transfer to Adam

For SGD the update is `−lr·g` — *linear* in the gradient. Scaling `lr ∝ b` and holding the momentum
timescale fixed makes B/b small steps sum to one large step (the noise averages out across the sum).
This is the classic linear scaling rule.

Adam breaks this because of the `√v̂` normalization. Two ways to see it:

- **Per-step magnitude (noise-dominated).** `m̂ → ⟨g⟩` (noise averages out of the first moment),
  `√v̂ → √(σ²/b)`, so the update `≈ lr·⟨g⟩/√(σ²/b) = lr·⟨g⟩·√b/σ ∝ lr·√b`. The batch enters as
  **√b**, not b — so a *linear* lr correction over-corrects.

- **"32 small steps ≠ one large step."** Each small step is `lr·m̂/√v̂`, and that factor is itself
  smaller for small batches: `m̂/√v̂ = ⟨g⟩/√(⟨g⟩²+σ²/b)`, whose denominator carries a noise floor
  `σ²/b` that is **larger** for small batches and is *not* removed by averaging longer (averaging
  smooths the estimate of `v̂`, not its mean). So `32` steps of `lr/32` sum to *less* than one large
  step — by a factor `√[(⟨g⟩²+σ²/B)/(⟨g⟩²+σ²/b)]`, which in the noise-dominated limit is `≈ 1/√(B/b)`.
  (In the *low-noise* limit the factor → 1 and the SGD intuition would hold; real small-batch training
  is in the noisy regime where it doesn't.)

This is exactly why a linearly-lr-scaled small-batch arm undertrains relative to large batch, while
keeping lr fixed at small batch *beats* large batch (more full-size — if noise-damped — steps per token).

## The square-root (SDE) scaling rule

The principled rule comes from the SDE view. Malladi et al.,
[*On the SDEs and Scaling Rules for Adaptive Gradient Algorithms*](https://arxiv.org/abs/2205.10287)
(NeurIPS 2022), derive an SDE for Adam/RMSprop and prove the discrete updates are a **1st-order weak
approximation** of it. The scaling rule that keeps all batch sizes as 1st-order weak approximations of
the **same SDE** — i.e. makes the trajectories provably coincide to that order — is, for batch scaled
by a factor δ:

```
lr  → lr · √δ
β1  → 1 − δ·(1 − β1)
β2  → 1 − δ·(1 − β2)
```

So learning rate scales as **√batch** (not linearly), and the EMA timescales are held fixed in
samples. This is the conservative "scale everything" instinct, corrected: the momentum scaling is
right, but the lr correction is √, not linear.

## Equivalence vs optimality — and why equivalence is what generalizes

There are two different questions, with different answers:

- **What lr is *optimal* at each batch size?** Empirically ~batch-*insensitive* for Adam — a very weak
  positive dependence (Marek et al., [arXiv:2507.07101](https://arxiv.org/abs/2507.07101), report the
  optimal lr moving only ~3× over a 1024× batch change). So "keep lr, scale only β2" is roughly optimal,
  and small batch then *beats* large batch on tokens-per-step efficiency.

- **What scaling makes small batch *reproduce* large batch?** The √ rule above (SDE equivalence).

The optimality answer is empirical and regime-specific — there's no guarantee it survives to late
training or transfers to RL. The **equivalence** answer carries a guarantee that is a property of the
*optimizer's response to gradient noise*, independent of the loss landscape: if small batch provably
reproduces the large-batch trajectory, then anything validated at large batch transfers. That is the
property we want "before optimizing."

**Caveats on the guarantee (honest scope):**
- It is a *1st-order weak* / small-lr SDE result; it degrades when lr is large enough that
  discretization error dominates.
- It guarantees equivalence *to large batch* — it inherits large batch's behavior, including any
  suboptimality. (Reproducing large batch is the point: safe and transferable, not maximal.)
- It assumes the gradient-noise / diffusion regime (below the critical batch size).
- For **RL**: the guarantee covers the optimizer dynamics (the task-agnostic part). It does **not**
  cover RL's non-stationarity (policy shift, off-policy staleness) — that is outside any batch-scaling
  SDE. But it is still the most defensible transfer basis, and it is tuning-free.

## Two effects that distinguish small from large batch even under the rule

These are the reasons we expect the √-rule small-batch run to be *close to but not identical* to large
batch, and they are hypotheses the experiment tests (not established results):

- **Sub-step freshness / staleness (favours small batch, slightly).** The SDE equivalence is only
  1st-order; the leading finite-step correction favours more, smaller steps, because each sub-step uses
  a gradient at a *fresher* iterate (less discretization "self-staleness"). A large step computes one
  gradient at `θ_t`; the equivalent small-batch sequence computes gradients at `θ_t, θ_{t+1}, …`. In RL
  this is sharper: staleness is a fixed ~1 generation regardless of step size, so taking many small
  sub-steps amortizes that fixed cost — small batch is structurally better-positioned on staleness.

- **Constant-lr noise floor (the "head start may not hold" worry).** With constant lr, training
  plateaus at a noise floor set by the SDE's stationary temperature, which grows with the *effective*
  lr. Keeping full lr at small batch (the "optimal"/paper setting) raises the temperature → a **higher
  floor**; the √-rule reproduces large batch's temperature → its floor. So full-lr small batch is
  expected to descend fastest early but settle higher, while the √-rule run tracks large batch and can
  overtake it late. (Counter-consideration: at leading order the displacement per token `∝ lr/√b`
  persists late, so whether the extra motion is progress or just noise around a higher floor is the
  empirical question — which is why the runs must go long.)

- **Higher-order correction.** Carrying the next term, `m̂/√v̂ = ⟨g⟩/√(⟨g⟩²+σ²/b) ≈
  (⟨g⟩/√(σ²/b))·(1 − ⟨g⟩²/2·b/σ²·…)`, gives an SNR-dependent correction (not a power law) that is
  largest when signal ≈ noise (early/mid training) and vanishes as noise dominates (late). So the
  √-rule is the asymptotically exact part; deviations live in the transient.

## Experiment

All arms branch from one lightly-pretrained checkpoint and share one fresh, disjoint token stream
(paired comparison); loss is compared vs **tokens seen** (`training/consumed_tokens`), since optimizer
steps are not comparable across batch sizes. Small batch b=16, large B=512 (ratio 32).

| Arm | batch | lr | β1 | β2 | what it isolates |
|---|---|---|---|---|---|
| A | 512 | 1e-4 | 0.9 | 0.95 | large-batch reference |
| B | 16 | 1e-4 | 0.9 | 0.95^(1/32) | keep-lr, β2 scaled (≈ empirical optimum) |
| D | 16 | 1e-4 | 0.9 | 0.95 | nothing scaled (naive) |
| G | 16 | 1e-4/√32 | 0.9 | 0.95 | √-lr only, **no** β scaling |
| I | 16 | 1e-4/√32 | 0.9 | 0.95^(1/32) | √-lr + β2 scaled |
| H | 16 | 1e-4/√32 | 0.9^(1/32) | 0.95^(1/32) | **√-lr + β1&β2 = full SDE rule** |

(Plus precision twins of A at b=512: TF32 and fp16.) G→I→H is a momentum-scaling ladder at the correct
(√) lr; H is the theoretically-guaranteed rule.

**Predictions.** H ≈ A across training (the equivalence guarantee), perhaps a hair better
(sub-step freshness). B fastest early but may settle to a higher floor late. D and the linear-lr
scalings undertrain. Precision twins overlap A (precision is not the axis that matters here).

**The two curves that decide it:** (1) does H track A *into late training*? (2) does B's early lead
over A/H *erode* late? The first is the equivalence guarantee; the second is whether the empirically-
optimal "keep lr" is a transient head start or a durable win.

## Results (preliminary — runs ongoing)

The experiment ran in two rounds, and the decisive lesson is about *regime*: **the √-rule holds in
the noise-dominated regime it is derived for, and that regime is only reached well into training.**

### Round 1 — signal-dominated, outside the rule's domain

Branching the arms early (loss ~5 → ~3 over the run) put them in a **signal-dominated** regime:
per-step gradients are dominated by signal, not noise. Three things followed, all observed:

- The √-rule is not even *expected* to apply — it is a diffusion / noise-dominated approximation.
- The noise-control knobs (batch beyond the critical batch, the β EMA windows, precision) had little
  leverage and washed out.
- The dominant effect was just the learning rate: the large-batch reference at 1e-4 was visibly
  *under*-tuned (a b=512 lr sweep 1e-4 → 3e-4 → 5.66e-4 closed most of the gap to the small-batch arms).

So round 1's batch-size differences were an update-count / drift effect, not the noise-averaging the
√-rule addresses — it was not a valid test of the equivalence. This motivated round 2.

### Round 2 — noise-dominated, the rule's domain

Re-anchored deep in training: arms branched (weights only, cold optimizer) from a checkpoint at
~18B tokens / loss ~2.8, batch ratio 16 (b=32 ↔ b=512 = 65,536 ↔ 1,048,576 tokens), keeping the 1e-4
baseline. Here the model is **noise-dominated** — measured per-step loss std is **0.027 at b=32 vs
0.008 at b=512, ratio ≈ 3.4** (close to the √16 = 4 that the 1/√batch noise law predicts), and this
per-step noise (~0.01–0.03) dwarfs the systematic inter-arm differences (~0.005) by an order of
magnitude. The std depends only on batch size, not on the optimizer settings.

In this regime the √/SDE rule reproduces the large-batch trajectory. Comparing at matched tokens
(loss-vs-tokens, binned over ~1B-token windows), the two √-rule pairs overlay:

| operating point | b=512 arm | b=32 (√-scaled) arm | gap |
|---|---|---|---|
| baseline (lr 1e-4) | A | H (lr 2.5e-5, β1 & β2 scaled) | ~0.0002 |
| aggressive (b=32 keep-lr) | J (lr 4e-4, √-up of B) | B (lr 1e-4, β2 scaled) | ~0.0006 |

Both gaps are ~10× smaller than the ~0.005 spread *between* the conservative (A/H) and aggressive
(B/J) operating points — i.e. the √-rule maps each batch size onto the other's trajectory at both
operating points.

**Secondary signals** — small (~0.0005–0.0008 nat) but directionally consistent at every bin, so
credible though not yet conclusive:

- **β1 scaling helps.** The full-SDE arms (both β scaled) sit slightly below the β2-only arms at both
  lr settings; dropping β1 makes the √-rule arm slightly *under*-track the baseline. A (mild) point for
  the full SDE rule over the paper "β2 only" rule.
- **fp16 edges bf16** by ~0.0005 (plausibly fp16's 10 mantissa bits vs bf16's 7) — refining "precision
  doesn't matter" to "precision barely matters, in the direction of more mantissa bits."
- **Batch size matters far less in round 2 than round 1** once scaled — consistent with the equivalence
  holding only in the noise-dominated regime.

**Regime is the through-line.** Round 1 (signal-dominated) is outside the rule's domain and the noise
knobs are inert; round 2 (noise-dominated) is inside it, the rule holds, and the noise knobs acquire
leverage. The knobs lighting up only in round 2 is itself a fingerprint of the regime. This is the
point for the motivation: **RL and late-stage training are noise-dominated — the same regime as round
2** — so "equivalence holds in round 2" is the evidence it transfers to the settings we care about.

### Methodology note: don't reuse the data seed when branching

Round 2's arms showed a sharp, persistent loss step at *exactly* 17.69B tokens. It is **not** a
data-quality artifact: the document shuffle is seeded independently of batch size / num_samples, so a
child run with the same seed replays the parent's document order. The arms branched from a checkpoint
trained to 17.69B tokens, so for the first 17.69B tokens they re-traverse data the parent already fit
(low loss), then jump to the true level on reaching novel data. **Lesson:** when branching from a
checkpoint, use a different data seed (or skip the parent's consumed prefix). All comparisons above are
windowed to the post-17.69B novel-data region.

### Status & caveats

- Runs are **ongoing and not converged**; numbers are preliminary and expected to firm up.
- Comparisons use **training loss**, not validation: the loss evaluator currently logs 0
  ([#538](https://github.com/ServiceNow/Fast-LLM/issues/538)). Training loss is a valid *relative*
  comparison here because all data features are shared across arms (paired design).
- Single 0.5B model, one dataset; the secondary signals are ~1.5σ per bin individually, credible only
  via cross-bin directional consistency.

## References

- McCandlish, Kaplan, Amodei et al., *An Empirical Model of Large-Batch Training*, [arXiv:1812.06162](https://arxiv.org/abs/1812.06162) — critical batch size / gradient noise scale.
- Malladi, Lyu, Panigrahi, Arora, *On the SDEs and Scaling Rules for Adaptive Gradient Algorithms*, [arXiv:2205.10287](https://arxiv.org/abs/2205.10287) (NeurIPS 2022) — square-root scaling rule with an SDE-equivalence guarantee.
- Marek et al., *(batch-size hyperparameter scaling)*, [arXiv:2507.07101](https://arxiv.org/abs/2507.07101) — empirical: optimal Adam lr is ~batch-insensitive; β2 scales to hold the token half-life.
