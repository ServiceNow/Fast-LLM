"""Per-loss vs monolithic (fused) head-loss benchmark: time + peak memory, individual losses + combos."""

import argparse

import torch
import torch._dynamo

from fast_llm.functional.config import EntropyLossType, TargetFormat
from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward
from fast_llm.layers.language_model.loss.monolithic import MonolithicLossSpec, monolithic_head_loss_forward_backward
from fast_llm.layers.language_model.loss.policy_gradient import compute_grpo_metrics, fused_grpo_loss_forward_backward
from fast_llm.layers.language_model.loss.z_loss import fused_z_loss_forward_backward

# Each scenario is a distinct fixed loss config (one compiled guard). The benchmark cycles many in one
# process; bump the limit so none falls back to eager. A real run has ONE config → compiles once.
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.cache_size_limit = 64


parser = argparse.ArgumentParser()
parser.add_argument("--num-tokens", type=int, default=16384)
parser.add_argument("--vocab", type=int, default=128256)
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--warmup", type=int, default=5)
args = parser.parse_args()

device = "cuda"
DTYPE = torch.bfloat16
GRAD = 1.0
N, V = args.num_tokens, args.vocab
torch.manual_seed(0)

logits = torch.randn(N, V, dtype=DTYPE, device=device)
labels = torch.randint(0, V, (N,), dtype=torch.int64, device=device)
teacher = torch.randn(N, V, dtype=DTYPE, device=device)
advantages = torch.randn(N, dtype=torch.float32, device=device)
old_logprobs = torch.randn(N, dtype=torch.float32, device=device)
num_labels_in_seq = torch.full((N,), N, dtype=torch.int32, device=device)
label_counts = torch.full((N,), float(N), dtype=torch.float32, device=device)
DIV = float(N)


# ---- per-loss parts (mirror the head's per_loss loop: each its own kernel/softmax, threading the grad) ----
def part_ce(grad):
    _, grad = fused_entropy_loss_forward_backward(
        logits=logits,
        target=labels,
        loss_mask=None,
        grad_logits=grad,
        grad_output=GRAD,
        group=None,
        logits_scale_factor=1.0,
        target_format=TargetFormat.labels,
        entropy_loss_type=EntropyLossType.cross_entropy,
        divisor=DIV,
    )
    return grad


def part_z(grad):
    _, grad = fused_z_loss_forward_backward(
        logits=logits,
        loss_mask=None,
        grad_logits=grad,
        grad_output=GRAD,
        group=None,
        logits_scale_factor=1.0,
        divisor=DIV,
    )
    return grad


def part_distill(grad):
    _, grad = fused_entropy_loss_forward_backward(
        logits=logits,
        target=teacher,
        loss_mask=None,
        grad_logits=grad,
        grad_output=GRAD,
        group=None,
        logits_scale_factor=1.0,
        target_format=TargetFormat.logits,
        entropy_loss_type=EntropyLossType.cross_entropy,
        divisor=DIV,
    )
    return grad


def part_grpo(grad):
    _, grad, _ = fused_grpo_loss_forward_backward(
        logits,
        labels,
        advantages,
        old_logprobs,
        grad_logits=grad,
        grad_output=GRAD,
        group=None,
        logits_scale_factor=1.0,
        num_labels_in_seq=num_labels_in_seq,
        divisor=DIV,
    )
    return grad


def part_grpo_metrics(grad):
    # The #494 second softmax: the per_loss path recomputes the softmax to get the metric family.
    compute_grpo_metrics(
        logits,
        labels,
        advantages,
        old_logprobs,
        label_counts,
        epsilon_low=0.2,
        epsilon_high=0.2,
        logits_scale_factor=1.0,
        group=None,
        compute_entropy=False,
    )
    return grad


def per_loss(parts):
    def run():
        grad = None
        for part in parts:
            grad = part(grad)
        return grad

    return run


# ---- fused specs ----
def spec_ce():
    return MonolithicLossSpec("cross_entropy", "ce", 1.0, 1.0, GRAD, DIV, target=labels)


def spec_z():
    return MonolithicLossSpec("z_loss", "z", 1.0, 1.0, GRAD, DIV)


def spec_distill():
    return MonolithicLossSpec(
        "entropy_from_distribution",
        "distill",
        1.0,
        1.0,
        GRAD,
        DIV,
        target=teacher,
        target_format=TargetFormat.logits,
        entropy_loss_type=EntropyLossType.cross_entropy,
    )


def spec_grpo(metrics=False):
    return MonolithicLossSpec(
        "grpo",
        "grpo",
        1.0,
        1.0,
        GRAD,
        DIV,
        target=labels,
        advantages=advantages,
        old_log_probabilities=old_logprobs,
        num_labels_in_seq=num_labels_in_seq,
        compute_metrics=metrics,
    )


def fused(specs):
    def run():
        return monolithic_head_loss_forward_backward(logits, specs, group=None, grad_logits=None)

    return run


def bench(run):
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], (torch.cuda.max_memory_allocated() - before) / 1e9


scenarios = [
    ("CE", [part_ce], [spec_ce()]),
    ("z_loss", [part_z], [spec_z()]),
    ("distillation", [part_distill], [spec_distill()]),
    ("GRPO", [part_grpo], [spec_grpo()]),
    ("CE+z", [part_ce, part_z], [spec_ce(), spec_z()]),
    ("CE+distill", [part_ce, part_distill], [spec_ce(), spec_distill()]),
    ("CE+z+distill", [part_ce, part_z, part_distill], [spec_ce(), spec_z(), spec_distill()]),
    ("GRPO+metrics", [part_grpo, part_grpo_metrics], [spec_grpo(metrics=True)]),
    ("GRPO+metrics+z", [part_grpo, part_grpo_metrics, part_z], [spec_grpo(metrics=True), spec_z()]),
]

print(f"shape: num_tokens={N} vocab={V} dtype={DTYPE} | iters={args.iters}")
print(
    f"{'scenario':<18}{'per_loss ms':>12}{'fused ms':>11}{'speedup':>9}{'per_loss GB':>13}{'fused GB':>10}{'mem saved':>11}"
)
for name, parts, specs in scenarios:
    torch.cuda.empty_cache()
    t_pl, m_pl = bench(per_loss(parts))
    torch.cuda.empty_cache()
    t_f, m_f = bench(fused(specs))
    print(f"{name:<18}{t_pl:>12.2f}{t_f:>11.2f}{t_pl / t_f:>8.2f}x{m_pl:>13.2f}{m_f:>10.2f}{(m_pl - m_f):>10.2f}G")
