import abc
import logging
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.config import CrossEntropyImpl, TargetFormat, TritonConfig
from fast_llm.layers.language_model.kwargs import LanguageModelKwargs, TargetsKwargs
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


@config_class(registry=True)
class LanguageModelLossConfig(Config):
    """
    Losses can register themselves using @config_class(dynamic_type= {LanguageModelLossConfig: "loss_type_name"}).
    """

    _name: typing.ClassVar[str]
    _abstract: typing.ClassVar[bool] = True

    weight: float = Field(
        default=1.0,
        hint=FieldHint.core,
        desc="Weight for this loss in the total loss computation.",
        valid=check_field(Assert.geq, 0.0),
    )

    distillation_model: str | None = Field(
        default=None,
        desc="Name of the reference model to use for knowledge distillation."
        "If provided, replace the loss with a distillation loss.",
        hint=FieldHint.feature,
    )

    @abc.abstractmethod
    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        pass

    def get_loss_definitions(self, name: str, count: int = 1, prediction_distance: int | None = None) -> LossDef:
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

    def get_formatted_name(self, registered_loss_name=None, prediction_distance: int | None = None) -> str:
        """
        Retruns loss name for logging as '<registered_loss_name>(<self._name>)', e.g. lm_loss(CE_loss), distillation(FwdKL_loss)
        """
        name = f"{registered_loss_name}({self._name})"
        if prediction_distance is not None:
            name = f"{name}_{prediction_distance}"
        return name

    @abc.abstractmethod
    def get_targets(
        self,
        kwargs: dict | None = None,
        prediction_distance: int | None = None,
        prediction_heads: int | None = None,
        sequence_parallel_logits: bool | None = None,
        group: "ProcessGroup" = None,
    ) -> dict[str, "torch.Tensor"]:
        pass


@config_class(dynamic_type={LanguageModelLossConfig: "cross_entropy"})
class CrossEntropyLMLossConfig(LanguageModelLossConfig):
    _name: typing.ClassVar[str] = "CE_loss"
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

    def get_targets(
        self,
        kwargs: dict | None = None,
        prediction_distance: int | None = None,
        prediction_heads: int | None = None,
        sequence_parallel_logits: bool | None = None,
        group: "ProcessGroup" = None,
    ) -> dict[str, "torch.Tensor"]:
        if kwargs is None:
            kwargs = {}

        lm_target = kwargs.get(LanguageModelKwargs.labels)
        if lm_target is not None:
            # MTP: Shift the labels
            lm_target_sequence_length = (
                lm_target.size(1 - kwargs[LanguageModelKwargs.sequence_first]) + 1 - prediction_heads
            )
            if LanguageModelKwargs.sequence_q_dim in kwargs:
                Assert.eq(lm_target_sequence_length, kwargs[LanguageModelKwargs.sequence_q_dim].size)
            lm_target_slice = slice(prediction_distance, prediction_distance + lm_target_sequence_length)
            lm_target = (
                lm_target[lm_target_slice]
                if kwargs[LanguageModelKwargs.sequence_first]
                else lm_target[:, lm_target_slice]
            ).flatten()
            if sequence_parallel_logits:
                from fast_llm.core.ops import split_op

                lm_target = split_op(lm_target, group, 0)
        return {TargetsKwargs.lm_target: lm_target}

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.cross_entropy import cross_entropy_forward_backward

        target = kwargs.get(TargetsKwargs.lm_target)
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

    _name: typing.ClassVar[str] = "FwdKL_loss"
    _abstract: typing.ClassVar[bool] = False

    teacher_softmax_temperature: float = Field(
        default=1.0,
        hint=FieldHint.optional,
        desc="Temperature for teacher softmax.",
        valid=check_field(Assert.gt, 0.0),
    )

    def _validate(self):
        assert self.distillation_model is not None, "Distillation loss required by ForwardKL Loss."
        super()._validate()

    def get_targets(
        self,
        kwargs: dict | None = None,
        prediction_distance: int | None = None,
        prediction_heads: int | None = None,
        sequence_parallel_logits: bool | None = None,
        group: "ProcessGroup" = None,
    ) -> dict[str, "torch.Tensor"]:
        if kwargs is None:
            kwargs = {}

        reference_model_logits = kwargs.get(f"{self.distillation_model}_logits")
        if reference_model_logits is not None:
            reference_model_logits = reference_model_logits.flatten(0, -2)
            if sequence_parallel_logits:
                from fast_llm.core.ops import split_op

                reference_model_logits = split_op(reference_model_logits, group, 0)
        return {TargetsKwargs.reference_model_logits: reference_model_logits}

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.cross_entropy import forward_kl_forward_backward

        target = kwargs.get(TargetsKwargs.reference_model_logits)

        return forward_kl_forward_backward(
            logits=logits.flatten(0, -2),
            target=target,
            loss_mask=loss_mask,
            grad_output=grad_output,
            group=group,
            logits_scale_factor=logits_scale_factor,
            teacher_softmax_temperature=self.teacher_softmax_temperature,
            target_format=TargetFormat.logits,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "reverse_kl_distillation"})
class ReverseKLLossConfig(ForwardKLLossConfig):
    """Reverse KL divergence KL(q||p) for distillation (mode-seeking)."""

    _name: typing.ClassVar[str] = "RevKL_loss"
    _abstract: typing.ClassVar[bool] = False

    def _validate(self):
        assert self.distillation_model is not None, "Distillation loss required by Reverse KL Loss."
        super()._validate()

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.cross_entropy import reverse_kl_forward_backward

        # Use distillation_target for KL losses
        target = kwargs.get(TargetsKwargs.reference_model_logits)

        return reverse_kl_forward_backward(
            logits=logits.flatten(0, -2),
            target=target,
            loss_mask=loss_mask,
            grad_output=grad_output,
            group=group,
            logits_scale_factor=logits_scale_factor,
            teacher_softmax_temperature=self.teacher_softmax_temperature,
            target_format=TargetFormat.logits,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "dpo"})
class DPOLossConfig(LanguageModelLossConfig):
    """Direct Preference Optimization (DPO) loss for alignment."""

    _name: typing.ClassVar[str] = "DPO_loss"
    _abstract: typing.ClassVar[bool] = False

    beta: float = Field(
        default=1.0,
        hint=FieldHint.core,
        desc="Beta parameter for DPO loss (controls strength of preference optimization).",
        valid=check_field(Assert.gt, 0.0),
    )

    dpo_reference_model: str | None = Field(
        default=None,
        desc="Name of the reference model to use for dpo.",
        hint=FieldHint.feature,
    )

    def _validate(self):
        assert self.dpo_reference_model is not None, "DPO loss requires a reference model."
        super()._validate()

    def get_targets(
        self,
        kwargs: dict | None = None,
        prediction_distance: int | None = None,
        prediction_heads: int | None = None,
        sequence_parallel_logits: bool | None = None,
        group: "ProcessGroup" = None,
    ) -> dict[str, "torch.Tensor"]:
        if kwargs is None:
            kwargs = {}

        reference_model_logits = kwargs.get(f"{self.dpo_reference_model}_logits")
        dpo_target = kwargs.get(LanguageModelKwargs.labels)
        if reference_model_logits is not None or dpo_target is not None:
            from fast_llm.core.ops import split_op

            if reference_model_logits is not None:
                reference_model_logits = reference_model_logits.flatten(0, -2)
                if sequence_parallel_logits:
                    reference_model_logits = split_op(reference_model_logits, group, 0)
            if dpo_target is not None:
                dpo_target = split_op(dpo_target, group, 0)
        return {
            TargetsKwargs.dpo_reference_model_logits: reference_model_logits,
            TargetsKwargs.dpo_target: dpo_target,
        }

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        group: "ProcessGroup" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.dpo import compute_dpo_loss

        dpo_target = kwargs.get(TargetsKwargs.dpo_target)
        dpo_reference_model_logits = kwargs.get(TargetsKwargs.dpo_reference_model_logits)
        chosen_spans = kwargs.get(LanguageModelKwargs.chosen_spans)
        rejected_spans = kwargs.get(LanguageModelKwargs.rejected_spans)

        return compute_dpo_loss(
            logits=logits,
            targets=dpo_target,
            reference_model_logits=dpo_reference_model_logits,
            chosen_spans=chosen_spans,
            rejected_spans=rejected_spans,
            beta=self.beta,
            grad_output=grad_output,
        )
