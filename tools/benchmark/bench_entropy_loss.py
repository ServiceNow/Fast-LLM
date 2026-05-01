"""
Benchmark entropy loss kernels.

All Triton kernels fuse fwd+bwd into a single logits-tensor pass; `grad_output=1.0`
triggers gradient computation alongside the loss.

Three main training cases benchmarked:

  cross_entropy + labels   — standard LM training (integer targets)
  cross_entropy + logits   — distillation CE with soft targets, p=softmax(target_logits)
  reverse_kl    + logits   — reverse KL divergence KL(q||p), p=softmax(target_logits)

z_loss is also included (shared input structure with the labels case).

Shapes fix tokens=4096, sweep vocab size from Llama-2 (32K) to Llama-3 (128K).
"""

import torch
import torch.nn.functional as F

from fast_llm.functional.config import EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from tools.benchmark.runner import Variant
from tools.benchmark.utils import bench_main, device, make_cases, standard_fwd_bwd_pytorch_variants

# (tokens, vocab_size)
_SHAPES = [
    (4096, 32768),  # 7B / Llama-2 vocab
    (4096, 65536),  # 64K vocab
    (4096, 131072),  # Llama-3 vocab
]
_DEFAULT_DTYPES = (torch.bfloat16,)


# --------------------------------------------------------------------------- inputs


def _make_label_inputs(tokens: int, vocab: int, dtype: torch.dtype) -> dict:
    return {
        "logits": torch.randn(tokens, vocab, dtype=dtype, device=device(), requires_grad=True),
        "labels": torch.randint(0, vocab, (tokens,), dtype=torch.long, device=device()),
    }


def _make_distribution_inputs(tokens: int, vocab: int, dtype: torch.dtype) -> dict:
    return {
        "logits": torch.randn(tokens, vocab, dtype=dtype, device=device(), requires_grad=True),
        # target_logits: teacher logits; no gradient needed w.r.t. these.
        "target_logits": torch.randn(tokens, vocab, dtype=dtype, device=device()),
    }


def _reset_logits_grad(inputs: dict) -> None:
    inputs["logits"].grad = None


# --------------------------------------------------------------------------- eager kernels


def _ce_labels_eager(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def _ce_dist_eager(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """CE(p, q) where p = softmax(target_logits), q = softmax(logits)."""
    return F.cross_entropy(logits, target_logits.softmax(dim=-1))


def _reverse_kl_eager(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """KL(q||p) where q = softmax(logits), p = softmax(target_logits)."""
    return F.kl_div(target_logits.log_softmax(dim=-1), logits.softmax(dim=-1), reduction="batchmean")


def _z_loss_eager(logits: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return (log_z * log_z).mean()


# --------------------------------------------------------------------------- variant assembly


def _entropy_variants(eager_function, input_keys, triton_kwargs=None) -> list[Variant]:
    variants = standard_fwd_bwd_pytorch_variants(
        eager_function,
        input_keys=input_keys,
        grad_input_keys=("logits",),
        output_key="loss",
        reset_inputs=_reset_logits_grad,
    )
    if TritonConfig.enabled():
        target_key = input_keys[1]
        kwargs = triton_kwargs or {}

        def triton_fwd(inputs: dict) -> dict:
            loss, _ = triton_entropy_loss_forward_backward(
                inputs["logits"], inputs[target_key], loss_mask=None, grad_output=None, **kwargs
            )
            return {"loss": loss}

        def triton_fwd_bwd(inputs: dict) -> dict:
            loss, grad_logits = triton_entropy_loss_forward_backward(
                inputs["logits"], inputs[target_key], loss_mask=None, grad_output=1.0, **kwargs
            )
            return {"loss": loss, "grad_logits": grad_logits}

        variants.append(Variant(name="fast_llm_triton", fwd=triton_fwd, fwd_bwd=triton_fwd_bwd))
    return variants


def _z_loss_variants() -> list[Variant]:
    variants = standard_fwd_bwd_pytorch_variants(
        _z_loss_eager,
        input_keys=("logits",),
        grad_input_keys=("logits",),
        output_key="loss",
        reset_inputs=_reset_logits_grad,
    )
    if TritonConfig.enabled():

        def triton_fwd(inputs: dict) -> dict:
            loss, _ = triton_z_loss_forward_backward(inputs["logits"], loss_mask=None, grad_output=None)
            return {"loss": loss}

        def triton_fwd_bwd(inputs: dict) -> dict:
            loss, grad_logits = triton_z_loss_forward_backward(inputs["logits"], loss_mask=None, grad_output=1.0)
            return {"loss": loss, "grad_logits": grad_logits}

        variants.append(Variant(name="fast_llm_triton", fwd=triton_fwd, fwd_bwd=triton_fwd_bwd))
    return variants


# --------------------------------------------------------------------------- bytes / flops


def _label_loss_bytes(tokens: int, vocab: int, dtype: torch.dtype) -> int:
    # fwd+bwd: read logits, read labels (int32), write grad_logits.
    return 2 * tokens * vocab * dtype.itemsize + tokens * 4


def _dist_loss_bytes(tokens: int, vocab: int, dtype: torch.dtype) -> int:
    # fwd+bwd: read logits, read target_logits, write grad_logits.
    return 3 * tokens * vocab * dtype.itemsize


def _entropy_loss_flops(tokens: int, vocab: int) -> int:
    # fwd ≈ 3*vocab per token (max, sum_exp, CE); bwd ≈ vocab. Total ≈ 4*vocab.
    return 4 * tokens * vocab


# --------------------------------------------------------------------------- entry point


def benchmarks(
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    shapes = shapes if shapes is not None else _SHAPES
    return [
        (
            "entropy_loss: cross_entropy (labels)",
            make_cases(
                "cross_entropy_labels", dtypes, shapes, _make_label_inputs, _label_loss_bytes, _entropy_loss_flops
            ),
            _entropy_variants(_ce_labels_eager, input_keys=("logits", "labels")),
        ),
        (
            "entropy_loss: cross_entropy (logits)",
            make_cases(
                "cross_entropy_logits",
                dtypes,
                shapes,
                _make_distribution_inputs,
                _dist_loss_bytes,
                _entropy_loss_flops,
            ),
            _entropy_variants(
                _ce_dist_eager,
                input_keys=("logits", "target_logits"),
                triton_kwargs={
                    "target_format": TargetFormat.logits,
                    "entropy_loss_type": EntropyLossType.cross_entropy,
                },
            ),
        ),
        (
            "entropy_loss: reverse_kl (logits)",
            make_cases(
                "reverse_kl_logits", dtypes, shapes, _make_distribution_inputs, _dist_loss_bytes, _entropy_loss_flops
            ),
            _entropy_variants(
                _reverse_kl_eager,
                input_keys=("logits", "target_logits"),
                triton_kwargs={
                    "target_format": TargetFormat.logits,
                    "entropy_loss_type": EntropyLossType.reverse_kl,
                },
            ),
        ),
        (
            "entropy_loss: z_loss",
            make_cases("z_loss", dtypes, shapes, _make_label_inputs, _label_loss_bytes, _entropy_loss_flops),
            _z_loss_variants(),
        ),
    ]


run = bench_main(benchmarks)


if __name__ == "__main__":
    run()
