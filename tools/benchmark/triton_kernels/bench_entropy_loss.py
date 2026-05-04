"""Cross-entropy and z-loss kernels: label-target CE, logit-target CE, reverse
KL, and z-loss (logsumexp²)."""

import dataclasses
import typing

import torch
import torch.nn.functional as F

from fast_llm.functional.config import EntropyLossType, TargetFormat, TritonConfig
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.functional.triton.z_loss import triton_z_loss_forward_backward
from tools.benchmark.triton_kernels.runner import DtypedCase, Inputs, Variant
from tools.benchmark.triton_kernels.utils import dtype_short, standard_pytorch_variants

# (tokens, vocab_size)
_SHAPES = [
    (4096, 32768),  # Llama-2 vocab
    (4096, 65536),
    (4096, 131072),  # Llama-3 vocab
]


@dataclasses.dataclass
class _EntropyCase(DtypedCase):
    tokens: int
    vocab: int
    dtype: torch.dtype

    @property
    def name(self) -> str:
        return f"({self.tokens}, {self.vocab}) {dtype_short(self.dtype)}"

    @property
    def expected_flops(self) -> int:
        # fwd ≈ 3*vocab per token, bwd ≈ vocab.
        return 4 * self.tokens * self.vocab


@dataclasses.dataclass
class EntropyLabelCase(_EntropyCase):
    @property
    def expected_bytes(self) -> int:
        # 2× logits + small labels traffic.
        return 2 * self.tokens * self.vocab * self.dtype.itemsize + self.tokens * 4

    def make_inputs(self, device: torch.device) -> Inputs:
        return {
            "logits": torch.randn(self.tokens, self.vocab, dtype=self.dtype, device=device, requires_grad=True),
            "labels": torch.randint(0, self.vocab, (self.tokens,), dtype=torch.long, device=device),
        }


@dataclasses.dataclass
class EntropyDistCase(_EntropyCase):
    @property
    def expected_bytes(self) -> int:
        # 2× logits + 1× target_logits.
        return 3 * self.tokens * self.vocab * self.dtype.itemsize

    def make_inputs(self, device: torch.device) -> Inputs:
        return {
            "logits": torch.randn(self.tokens, self.vocab, dtype=self.dtype, device=device, requires_grad=True),
            "target_logits": torch.randn(self.tokens, self.vocab, dtype=self.dtype, device=device),
        }


def _ce_labels_eager(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def _ce_dist_eager(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target_logits.softmax(dim=-1))


def _reverse_kl_eager(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    return F.kl_div(target_logits.log_softmax(dim=-1), logits.softmax(dim=-1), reduction="batchmean")


def _z_loss_eager(logits: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits.float(), dim=-1)
    return (log_z * log_z).mean()


def _entropy_variants(
    eager_function: typing.Callable,
    input_keys: tuple[str, ...],
    triton_kwargs: dict | None = None,
) -> list[Variant]:
    """Variants for the 3 entropy_loss kernels that share `triton_entropy_loss_forward_backward`."""
    target_key = input_keys[1]
    triton_kwargs = triton_kwargs or {}

    def triton_fwd(inputs: Inputs) -> dict:
        loss, _ = triton_entropy_loss_forward_backward(
            inputs["logits"], inputs[target_key], loss_mask=None, grad_output=None, **triton_kwargs
        )
        return {"loss": loss}

    def triton_fwd_bwd(inputs: Inputs) -> dict:
        loss, grad_logits = triton_entropy_loss_forward_backward(
            inputs["logits"], inputs[target_key], loss_mask=None, grad_output=1.0, **triton_kwargs
        )
        return {"loss": loss, "grad_logits": grad_logits}

    variants = standard_pytorch_variants(
        eager_function,
        input_keys=input_keys,
        grad_input_keys=("logits",),
        output_key="loss",
    )
    if TritonConfig.enabled():
        variants.append(Variant(name="fast_llm_triton", fwd=triton_fwd, fwd_bwd=triton_fwd_bwd))
    return variants


def _z_loss_triton_fwd(inputs: Inputs) -> dict:
    loss, _ = triton_z_loss_forward_backward(inputs["logits"], loss_mask=None, grad_output=None)
    return {"loss": loss}


def _z_loss_triton_fwd_bwd(inputs: Inputs) -> dict:
    loss, grad_logits = triton_z_loss_forward_backward(inputs["logits"], loss_mask=None, grad_output=1.0)
    return {"loss": loss, "grad_logits": grad_logits}


def benchmarks(
    dtypes: tuple[torch.dtype, ...],
    shapes: list[tuple[int, int]] | None = None,
) -> list[tuple[str, list, list]]:
    shapes = shapes if shapes is not None else _SHAPES
    label_cases = [EntropyLabelCase(tokens=t, vocab=v, dtype=d) for d in dtypes for (t, v) in shapes]
    dist_cases = [EntropyDistCase(tokens=t, vocab=v, dtype=d) for d in dtypes for (t, v) in shapes]
    z_loss_variants = standard_pytorch_variants(
        _z_loss_eager,
        input_keys=("logits",),
        grad_input_keys=("logits",),
        output_key="loss",
    )
    if TritonConfig.enabled():
        z_loss_variants.append(Variant(name="fast_llm_triton", fwd=_z_loss_triton_fwd, fwd_bwd=_z_loss_triton_fwd_bwd))
    return [
        (
            "entropy_loss: cross_entropy (labels)",
            label_cases,
            _entropy_variants(_ce_labels_eager, input_keys=("logits", "labels")),
        ),
        (
            "entropy_loss: cross_entropy (logits)",
            dist_cases,
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
            dist_cases,
            _entropy_variants(
                _reverse_kl_eager,
                input_keys=("logits", "target_logits"),
                triton_kwargs={
                    "target_format": TargetFormat.logits,
                    "entropy_loss_type": EntropyLossType.reverse_kl,
                },
            ),
        ),
        ("entropy_loss: z_loss", label_cases, z_loss_variants),
    ]
