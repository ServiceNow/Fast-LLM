import abc
import dataclasses
import logging
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.config import CrossEntropyImpl, TargetFormat, TritonConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.core.distributed import ProcessGroup

logger = logging.getLogger(__name__)

#
# CE loss on lm_targets for standard LM training. Here targets are already masked.
# CE loss for distillation: cross entropuy that uses reference_model_logits as soft targets, not implemented, TODO.
# Forward KL divergence loss on reference_model_logits for distillation (mode-covering).
# Reverse KL divergence loss on reference_model_logits for distillation (mode-seeking).
# DPO loss for alignment using chosen and rejected spans.
#


def _format_name(name: str) -> str:
    return name.replace("_", " ")


@dataclasses.dataclass
class Targets:
    lm_target: "torch.Tensor | None" = None
    dpo_target: "torch.Tensor | None" = None
    loss_mask: "torch.Tensor | None" = None
    reference_model_logits: "torch.Tensor | None" = None
    dpo_reference_model_logits: "torch.Tensor | None" = None

    def has_any_target(self) -> bool:
        return any(getattr(self, field.name) is not None for field in dataclasses.fields(self))


@config_class(registry=True)
class LanguageModelLossConfig(Config):
    """
    Losses canm register themselves
    using @config_class(dynamic_type= {LanguageModelLossConfig: "loss_type_name"})
    """

    _name: typing.ClassVar[str]
    _abstract: typing.ClassVar[bool] = True

    weight: float = Field(
        default=1.0,
        hint=FieldHint.core,
        desc="Weight for this loss in the total loss computation.",
        valid=check_field(Assert.geq, 0.0),
    )

    @abc.abstractmethod
    def compute_loss(
        self,
        logits: "torch.Tensor",
        targets: Targets,
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        pass

    def get_loss_def(self, name: str, count: int = 1, prediction_distance: int | None = None) -> LossDef:
        name = self.get_formatted_name(name, prediction_distance)
        return LossDef(
            name=name,
            formatted_name=_format_name(name),
            count=count,
            dtype=DataType.float32,
        )

    def _validate(self):
        Assert.geq(self.weight, 0.0)
        super()._validate()

    def get_formatted_name(self, name=None, prediction_distance: int | None = None) -> str:
        name = f"{self._name}({name})"
        if prediction_distance is not None:
            name = f"{name}_{prediction_distance}"
        return name


@config_class(dynamic_type={LanguageModelLossConfig: "cross_entropy"})
class CrossEntropyLMLossConfig(LanguageModelLossConfig):
    _name: typing.ClassVar[str] = "CE"
    _abstract: typing.ClassVar[bool] = False

    implementation: CrossEntropyImpl = Field(
        default=CrossEntropyImpl.auto,
        desc="Implementation for the cross-entropy computation.",
        hint=FieldHint.performance,
    )

    teacher_softmax_temperature: float = Field(
        default=1.0,
        hint=FieldHint.optional,
        desc="Temperature for teacher softmax (used in distillation losses).",
        valid=check_field(Assert.gt, 0.0),
    )

    def compute_loss(
        self,
        logits: "torch.Tensor",
        targets: Targets,
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.cross_entropy import cross_entropy_forward_backward

        target = targets.lm_target
        if target is None:
            raise ValueError("CrossEntropyLoss requires lm_target to be set in Targets")
        implementation = self.implementation
        if implementation == CrossEntropyImpl.auto:
            if vocab_parallel:
                implementation = CrossEntropyImpl.fused
            elif TritonConfig.TRITON_ENABLED:
                implementation = CrossEntropyImpl.triton
            else:
                implementation = CrossEntropyImpl.fused

        return cross_entropy_forward_backward(
            logits=logits.flatten(0, -2),
            target=target,
            loss_mask=None,  # Labels are already masked
            grad_output=grad_output,
            group=group,
            implementation=implementation,
            logits_scale_factor=logits_scale_factor,
            teacher_softmax_temperature=self.teacher_softmax_temperature,
            target_format=TargetFormat.labels,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "forward_kl_distillation"})
class ForwardKLLossConfig(LanguageModelLossConfig):
    """Forward KL divergence KL(p||q) for distillation (mode-covering)."""

    _name: typing.ClassVar[str] = "FwdKL"
    _abstract: typing.ClassVar[bool] = False

    teacher_softmax_temperature: float = Field(
        default=1.0,
        hint=FieldHint.optional,
        desc="Temperature for teacher softmax.",
        valid=check_field(Assert.gt, 0.0),
    )

    def compute_loss(
        self,
        logits: "torch.Tensor",
        targets: Targets,
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.cross_entropy import forward_kl_forward_backward

        target = targets.reference_model_logits
        if target is None:
            raise ValueError("ForwardKLLoss requires distillation_target to be set in Targets")

        return forward_kl_forward_backward(
            logits=logits.flatten(0, -2),
            target=target,
            loss_mask=targets.loss_mask,
            grad_output=grad_output,
            group=group,
            logits_scale_factor=logits_scale_factor,
            teacher_softmax_temperature=self.teacher_softmax_temperature,
            target_format=TargetFormat.logits,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "reverse_kl_distillation"})
class ReverseKLLossConfig(LanguageModelLossConfig):
    """Reverse KL divergence KL(q||p) for distillation (mode-seeking)."""

    _name: typing.ClassVar[str] = "RevKL"
    _abstract: typing.ClassVar[bool] = False

    teacher_softmax_temperature: float = Field(
        default=1.0,
        hint=FieldHint.optional,
        desc="Temperature for teacher softmax.",
        valid=check_field(Assert.gt, 0.0),
    )

    def compute_loss(
        self,
        logits: "torch.Tensor",
        targets: Targets,
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.cross_entropy import reverse_kl_forward_backward

        # Use distillation_target for KL losses
        target = targets.reference_model_logits
        if target is None:
            raise ValueError("ReverseKLLoss requires distillation_target to be set in Targets")

        return reverse_kl_forward_backward(
            logits=logits.flatten(0, -2),
            target=target,
            loss_mask=targets.loss_mask,
            grad_output=grad_output,
            group=group,
            logits_scale_factor=logits_scale_factor,
            teacher_softmax_temperature=self.teacher_softmax_temperature,
            target_format=TargetFormat.logits,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "dpo"})
class DPOLossConfig(LanguageModelLossConfig):
    """Direct Preference Optimization (DPO) loss for alignment."""

    _name: typing.ClassVar[str] = "DPO"
    _abstract: typing.ClassVar[bool] = False

    beta: float = Field(
        default=1.0,
        hint=FieldHint.core,
        desc="Beta parameter for DPO loss (controls strength of preference optimization).",
        valid=check_field(Assert.gt, 0.0),
    )

    def compute_loss(
        self,
        logits: "torch.Tensor",
        targets: Targets,
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.dpo import compute_dpo_loss
        from fast_llm.layers.language_model.config import LanguageModelKwargs

        chosen_spans = kwargs.get(LanguageModelKwargs.chosen_spans)
        rejected_spans = kwargs.get(LanguageModelKwargs.rejected_spans)

        return compute_dpo_loss(
            logits=logits,
            targets=targets.dpo_target,
            reference_model_logits=targets.dpo_reference_model_logits,
            chosen_spans=chosen_spans,
            rejected_spans=rejected_spans,
            beta=self.beta,
            grad_output=grad_output,
        )
