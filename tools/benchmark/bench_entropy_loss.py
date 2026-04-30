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

from functools import partial

import torch
import torch.nn.functional as F

from fast_llm.functional.config import EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from tools.benchmark.runner import Case, Variant, run_benchmark
from tools.benchmark.utils import case_name, device

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


# --------------------------------------------------------------------------- cross_entropy (labels)


def _ce_labels_eager(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


_ce_labels_compiled_default = torch.compile(_ce_labels_eager, mode="default", dynamic=False)
_ce_labels_compiled_max = torch.compile(_ce_labels_eager, mode="max-autotune-no-cudagraphs", dynamic=False)


def _run_ce_labels_fwd(inputs: dict, fn) -> dict:
    return {"loss": fn(inputs["logits"], inputs["labels"])}


def _run_ce_labels_fwd_fp32(inputs: dict) -> dict:
    logits_fp32 = inputs["logits"].float().detach().requires_grad_(True)
    return {"loss": _ce_labels_eager(logits_fp32, inputs["labels"])}


def _reset_logits_grad(inputs: dict) -> None:
    inputs["logits"].grad = None


def _run_ce_labels_fwd_bwd(inputs: dict, fn) -> dict:
    loss = fn(inputs["logits"], inputs["labels"])
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": inputs["logits"].grad}


def _run_ce_labels_fwd_bwd_fp32(inputs: dict) -> dict:
    logits_fp32 = inputs["logits"].float().detach().requires_grad_(True)
    loss = _ce_labels_eager(logits_fp32, inputs["labels"])
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": logits_fp32.grad}


def _run_ce_labels_fwd_triton(inputs: dict) -> dict:
    loss, _ = triton_entropy_loss_forward_backward(
        inputs["logits"], inputs["labels"], loss_mask=None, grad_output=None
    )
    return {"loss": loss}


def _run_ce_labels_fwd_bwd_triton(inputs: dict) -> dict:
    loss, grad_logits = triton_entropy_loss_forward_backward(
        inputs["logits"], inputs["labels"], loss_mask=None, grad_output=1.0
    )
    return {"loss": loss, "grad_logits": grad_logits}


def _ce_labels_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=_run_ce_labels_fwd_fp32,
            fwd_bwd=_run_ce_labels_fwd_bwd_fp32,
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inputs: _run_ce_labels_fwd(inputs, _ce_labels_eager),
            fwd_bwd=lambda inputs: _run_ce_labels_fwd_bwd(inputs, _ce_labels_eager),
            reset_inputs=_reset_logits_grad,
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: _run_ce_labels_fwd(inputs, _ce_labels_compiled_default),
            fwd_bwd=lambda inputs: _run_ce_labels_fwd_bwd(inputs, _ce_labels_compiled_default),
            reset_inputs=_reset_logits_grad,
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inputs: _run_ce_labels_fwd(inputs, _ce_labels_compiled_max),
            fwd_bwd=lambda inputs: _run_ce_labels_fwd_bwd(inputs, _ce_labels_compiled_max),
            reset_inputs=_reset_logits_grad,
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_run_ce_labels_fwd_triton,
                fwd_bwd=_run_ce_labels_fwd_bwd_triton,
            )
        )
    return variants


# --------------------------------------------------------------------------- cross_entropy (logits / distribution)


def _ce_dist_eager(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """CE(p, q) where p = softmax(target_logits), q = softmax(logits)."""
    return F.cross_entropy(logits, target_logits.softmax(dim=-1))


_ce_dist_compiled_default = torch.compile(_ce_dist_eager, mode="default", dynamic=False)
_ce_dist_compiled_max = torch.compile(_ce_dist_eager, mode="max-autotune-no-cudagraphs", dynamic=False)


def _run_dist_fwd(inputs: dict, fn) -> dict:
    return {"loss": fn(inputs["logits"], inputs["target_logits"])}


def _run_ce_dist_fwd_fp32(inputs: dict) -> dict:
    logits_fp32 = inputs["logits"].float().detach().requires_grad_(True)
    return {"loss": _ce_dist_eager(logits_fp32, inputs["target_logits"].float())}


def _run_dist_fwd_bwd(inputs: dict, fn) -> dict:
    loss = fn(inputs["logits"], inputs["target_logits"])
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": inputs["logits"].grad}


def _run_ce_dist_fwd_bwd_fp32(inputs: dict) -> dict:
    logits_fp32 = inputs["logits"].float().detach().requires_grad_(True)
    loss = _ce_dist_eager(logits_fp32, inputs["target_logits"].float())
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": logits_fp32.grad}


def _run_ce_dist_fwd_triton(inputs: dict) -> dict:
    loss, _ = triton_entropy_loss_forward_backward(
        inputs["logits"],
        inputs["target_logits"],
        loss_mask=None,
        grad_output=None,
        target_format=TargetFormat.logits,
        entropy_loss_type=EntropyLossType.cross_entropy,
    )
    return {"loss": loss}


def _run_ce_dist_fwd_bwd_triton(inputs: dict) -> dict:
    loss, grad_logits = triton_entropy_loss_forward_backward(
        inputs["logits"],
        inputs["target_logits"],
        loss_mask=None,
        grad_output=1.0,
        target_format=TargetFormat.logits,
        entropy_loss_type=EntropyLossType.cross_entropy,
    )
    return {"loss": loss, "grad_logits": grad_logits}


def _ce_dist_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=_run_ce_dist_fwd_fp32,
            fwd_bwd=_run_ce_dist_fwd_bwd_fp32,
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inputs: _run_dist_fwd(inputs, _ce_dist_eager),
            fwd_bwd=lambda inputs: _run_dist_fwd_bwd(inputs, _ce_dist_eager),
            reset_inputs=_reset_logits_grad,
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: _run_dist_fwd(inputs, _ce_dist_compiled_default),
            fwd_bwd=lambda inputs: _run_dist_fwd_bwd(inputs, _ce_dist_compiled_default),
            reset_inputs=_reset_logits_grad,
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inputs: _run_dist_fwd(inputs, _ce_dist_compiled_max),
            fwd_bwd=lambda inputs: _run_dist_fwd_bwd(inputs, _ce_dist_compiled_max),
            reset_inputs=_reset_logits_grad,
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_run_ce_dist_fwd_triton,
                fwd_bwd=_run_ce_dist_fwd_bwd_triton,
            )
        )
    return variants


# --------------------------------------------------------------------------- reverse_kl (logits / distribution)


def _reverse_kl_eager(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """KL(q||p) where q = softmax(logits), p = softmax(target_logits)."""
    return F.kl_div(
        target_logits.log_softmax(dim=-1),
        logits.softmax(dim=-1),
        reduction="batchmean",
    )


_reverse_kl_compiled_default = torch.compile(_reverse_kl_eager, mode="default", dynamic=False)
_reverse_kl_compiled_max = torch.compile(_reverse_kl_eager, mode="max-autotune-no-cudagraphs", dynamic=False)


def _run_rkl_fwd_fp32(inputs: dict) -> dict:
    logits_fp32 = inputs["logits"].float().detach().requires_grad_(True)
    return {"loss": _reverse_kl_eager(logits_fp32, inputs["target_logits"].float())}


def _run_rkl_fwd_bwd_fp32(inputs: dict) -> dict:
    logits_fp32 = inputs["logits"].float().detach().requires_grad_(True)
    loss = _reverse_kl_eager(logits_fp32, inputs["target_logits"].float())
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": logits_fp32.grad}


def _run_rkl_fwd_triton(inputs: dict) -> dict:
    loss, _ = triton_entropy_loss_forward_backward(
        inputs["logits"],
        inputs["target_logits"],
        loss_mask=None,
        grad_output=None,
        target_format=TargetFormat.logits,
        entropy_loss_type=EntropyLossType.reverse_kl,
    )
    return {"loss": loss}


def _run_rkl_fwd_bwd_triton(inputs: dict) -> dict:
    loss, grad_logits = triton_entropy_loss_forward_backward(
        inputs["logits"],
        inputs["target_logits"],
        loss_mask=None,
        grad_output=1.0,
        target_format=TargetFormat.logits,
        entropy_loss_type=EntropyLossType.reverse_kl,
    )
    return {"loss": loss, "grad_logits": grad_logits}


def _reverse_kl_variants() -> list[Variant]:
    variants = [
        Variant(
            name="fp32_reference",
            fwd=_run_rkl_fwd_fp32,
            fwd_bwd=_run_rkl_fwd_bwd_fp32,
            is_reference=True,
        ),
        Variant(
            name="pytorch_eager",
            fwd=lambda inputs: _run_dist_fwd(inputs, _reverse_kl_eager),
            fwd_bwd=lambda inputs: _run_dist_fwd_bwd(inputs, _reverse_kl_eager),
            reset_inputs=_reset_logits_grad,
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: _run_dist_fwd(inputs, _reverse_kl_compiled_default),
            fwd_bwd=lambda inputs: _run_dist_fwd_bwd(inputs, _reverse_kl_compiled_default),
            reset_inputs=_reset_logits_grad,
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inputs: _run_dist_fwd(inputs, _reverse_kl_compiled_max),
            fwd_bwd=lambda inputs: _run_dist_fwd_bwd(inputs, _reverse_kl_compiled_max),
            reset_inputs=_reset_logits_grad,
        ),
    ]
    if TritonConfig.enabled():
        variants.append(
            Variant(
                name="fast_llm_triton",
                fwd=_run_rkl_fwd_triton,
                fwd_bwd=_run_rkl_fwd_bwd_triton,
            )
        )
    return variants


# --------------------------------------------------------------------------- z_loss


def _z_loss_eager(logits: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return (log_z * log_z).mean()


_z_loss_compiled_default = torch.compile(_z_loss_eager, mode="default", dynamic=False)
_z_loss_compiled_max = torch.compile(_z_loss_eager, mode="max-autotune-no-cudagraphs", dynamic=False)


def _run_zl_fwd(inputs: dict, fn) -> dict:
    return {"loss": fn(inputs["logits"])}


def _run_zl_fwd_fp32(inputs: dict) -> dict:
    logits_fp32 = inputs["logits"].float().detach().requires_grad_(True)
    return {"loss": _z_loss_eager(logits_fp32)}


def _run_zl_fwd_bwd(inputs: dict, fn) -> dict:
    loss = fn(inputs["logits"])
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": inputs["logits"].grad}


def _run_zl_fwd_bwd_fp32(inputs: dict) -> dict:
    logits_fp32 = inputs["logits"].float().detach().requires_grad_(True)
    loss = _z_loss_eager(logits_fp32)
    loss.backward()
    return {"loss": loss.detach(), "grad_logits": logits_fp32.grad}


def _run_zl_fwd_triton(inputs: dict) -> dict:
    loss, _ = triton_z_loss_forward_backward(inputs["logits"], loss_mask=None, grad_output=None)
    return {"loss": loss}


def _run_zl_fwd_bwd_triton(inputs: dict) -> dict:
    loss, grad_logits = triton_z_loss_forward_backward(inputs["logits"], loss_mask=None, grad_output=1.0)
    return {"loss": loss, "grad_logits": grad_logits}


def _z_loss_variants() -> list[Variant]:
    variants = [
        Variant(name="fp32_reference", fwd=_run_zl_fwd_fp32, fwd_bwd=_run_zl_fwd_bwd_fp32, is_reference=True),
        Variant(
            name="pytorch_eager",
            fwd=lambda inputs: _run_zl_fwd(inputs, _z_loss_eager),
            fwd_bwd=lambda inputs: _run_zl_fwd_bwd(inputs, _z_loss_eager),
            reset_inputs=_reset_logits_grad,
        ),
        Variant(
            name="pytorch_compiled",
            fwd=lambda inputs: _run_zl_fwd(inputs, _z_loss_compiled_default),
            fwd_bwd=lambda inputs: _run_zl_fwd_bwd(inputs, _z_loss_compiled_default),
            reset_inputs=_reset_logits_grad,
        ),
        Variant(
            name="pytorch_compiled_max",
            fwd=lambda inputs: _run_zl_fwd(inputs, _z_loss_compiled_max),
            fwd_bwd=lambda inputs: _run_zl_fwd_bwd(inputs, _z_loss_compiled_max),
            reset_inputs=_reset_logits_grad,
        ),
    ]
    if TritonConfig.enabled():
        variants.append(Variant(name="fast_llm_triton", fwd=_run_zl_fwd_triton, fwd_bwd=_run_zl_fwd_bwd_triton))
    return variants


# --------------------------------------------------------------------------- cases


def _bytes_per_element(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _label_loss_bytes(tokens: int, vocab: int, dtype: torch.dtype) -> int:
    """fwd+bwd: read logits, read labels (int32), write grad_logits."""
    element_size = _bytes_per_element(dtype)
    return 2 * tokens * vocab * element_size + tokens * 4


def _dist_loss_bytes(tokens: int, vocab: int, dtype: torch.dtype) -> int:
    """fwd+bwd: read logits, read target_logits, write grad_logits."""
    element_size = _bytes_per_element(dtype)
    return 3 * tokens * vocab * element_size


def _entropy_loss_flops(tokens: int, vocab: int) -> int:
    # fwd ≈ 3*vocab per token (max, sum_exp, CE); bwd ≈ vocab. Total ≈ 4*vocab.
    return 4 * tokens * vocab


def _label_cases(
    kernel_name: str, dtypes: tuple[torch.dtype, ...], shapes: list[tuple[int, int]] | None = None
) -> list[Case]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        Case(
            name=case_name(kernel_name, (tokens, vocab), dtype),
            make_inputs=partial(_make_label_inputs, tokens, vocab, dtype),
            expected_bytes=_label_loss_bytes(tokens, vocab, dtype),
            expected_flops=_entropy_loss_flops(tokens, vocab),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, vocab in shapes
    ]


def _dist_cases(
    kernel_name: str, dtypes: tuple[torch.dtype, ...], shapes: list[tuple[int, int]] | None = None
) -> list[Case]:
    shapes = shapes if shapes is not None else _SHAPES
    return [
        Case(
            name=case_name(kernel_name, (tokens, vocab), dtype),
            make_inputs=partial(_make_distribution_inputs, tokens, vocab, dtype),
            expected_bytes=_dist_loss_bytes(tokens, vocab, dtype),
            expected_flops=_entropy_loss_flops(tokens, vocab),
            compute_dtype=dtype,
        )
        for dtype in dtypes
        for tokens, vocab in shapes
    ]


# --------------------------------------------------------------------------- entry point


def run(
    verbose: bool = False,
    dtypes: tuple[torch.dtype, ...] | None = None,
    shapes: list[tuple[int, int]] | None = None,
) -> None:
    dtypes = tuple(dtypes) if dtypes else _DEFAULT_DTYPES
    run_benchmark(
        "entropy_loss: cross_entropy (labels)",
        _label_cases("cross_entropy_labels", dtypes, shapes),
        _ce_labels_variants(),
        verbose=verbose,
    )
    run_benchmark(
        "entropy_loss: cross_entropy (logits)",
        _dist_cases("cross_entropy_logits", dtypes, shapes),
        _ce_dist_variants(),
        verbose=verbose,
    )
    run_benchmark(
        "entropy_loss: reverse_kl (logits)",
        _dist_cases("reverse_kl_logits", dtypes, shapes),
        _reverse_kl_variants(),
        verbose=verbose,
    )
    run_benchmark("entropy_loss: z_loss", _label_cases("z_loss", dtypes, shapes), _z_loss_variants(), verbose=verbose)


if __name__ == "__main__":
    run()
