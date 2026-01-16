import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType, TargetFormat, TritonConfig
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import torch


class LanguageModelLossKwargs(BlockKwargs):
    labels = "labels"
    chosen_spans = "chosen_spans"
    rejected_spans = "rejected_spans"


@config_class(registry=True)
class LanguageModelLossConfig(Config):
    _abstract: typing.ClassVar[bool] = True

    weight: float = Field(
        default=1.0,
        hint=FieldHint.core,
        desc="Weight for this loss in the total loss computation.",
        valid=check_field(Assert.geq, 0.0),
    )

    def get_name(self, prediction_distance: int = 0) -> str:
        return self._name if prediction_distance == 0 else f"{self._name}_{prediction_distance}"

    @property
    def _name(self) -> str:
        raise NotImplementedError()

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        *,
        group: "torch.distributed.ProcessGroup|None" = None,
        logits_scale_factor: float = 1.0,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        split_index: int = 0,
        num_splits: int = 1,
        sequence_parallel_logits: bool = False,
        kwargs: dict[str, typing.Any],
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        raise NotImplementedError()


@config_class(dynamic_type={LanguageModelLossConfig: "label"})
class LanguageModelLabelEntropyLossConfig(LanguageModelLossConfig):
    _name: typing.ClassVar[str] = "CE_loss"
    _abstract: typing.ClassVar[bool] = False

    loss_type: EntropyLossType = Field(
        default=EntropyLossType.cross_entropy,
        desc="Type of loss to use.",
        hint=FieldHint.core,
    )

    implementation: EntropyLossImplementation = Field(
        default=EntropyLossImplementation.auto,
        desc="Loss implementation.",
        hint=FieldHint.performance,
    )

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        *,
        group: "torch.distributed.ProcessGroup|None" = None,
        logits_scale_factor: float = 1.0,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        split_index: int = 0,
        num_splits: int = 1,
        sequence_parallel_logits: bool = False,
        kwargs: dict[str, typing.Any],
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.entropy_loss import entropy_loss_forward_backward

        labels = kwargs[LanguageModelLossKwargs.labels]

        # MTP: Shift the labels
        if prediction_heads > 1:
            sequence_q_length = labels.size(1 - kwargs[LanguageModelLossKwargs.sequence_first]) + 1 - prediction_heads
            if LanguageModelLossKwargs.sequence_q_dim in kwargs:
                Assert.eq(sequence_q_length, kwargs[LanguageModelLossKwargs.sequence_q_dim].size)
            label_slice = slice(prediction_distance, prediction_distance + sequence_q_length)
            labels = labels[label_slice] if kwargs[LanguageModelLossKwargs.sequence_first] else labels[:, label_slice]

        labels = labels.flatten()

        # Get the local chunk.
        if sequence_parallel_logits:
            from fast_llm.core.ops import split_op

            labels = split_op(labels, group, 0)

        # Get the chunk for the current split.
        if num_splits > 1:
            labels = labels.chunk(num_splits)[split_index]

        implementation = self.implementation
        if implementation == EntropyLossImplementation.auto:
            if (
                TritonConfig.TRITON_ENABLED
                and torch.cuda.is_available()
                and group is None
                and self.loss_type == EntropyLossType.cross_entropy
            ):
                implementation = EntropyLossImplementation.triton
            else:
                implementation = EntropyLossImplementation.fused

        return entropy_loss_forward_backward(
            logits,
            labels,
            None,  # Labels are already masked
            grad_output=grad_output,
            group=group,
            implementation=implementation,
            logits_scale_factor=logits_scale_factor,
            target_format=TargetFormat.labels,
            entropy_loss_type=self.loss_type,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "distillation"})
class LanguageModelDistillationLossConfig(LanguageModelLossConfig):
    _name: typing.ClassVar[str] = "FwdKL_loss"
    _abstract: typing.ClassVar[bool] = False

    loss_type: EntropyLossType = Field(
        default=EntropyLossType.cross_entropy,
        desc="Type of loss to use.",
        hint=FieldHint.core,
    )
    implementation: EntropyLossImplementation = Field(
        default=EntropyLossImplementation.auto,
        desc="Loss implementation.",
        hint=FieldHint.performance,
    )
    reference_model: str = Field(
        desc="Name of the reference model for knowledge distillation.",
        hint=FieldHint.feature,
    )
    temperature: float = Field(
        default=1.0,
        hint=FieldHint.optional,
        desc="Temperature for teacher softmax.",
        valid=check_field(Assert.gt, 0.0),
    )

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        *,
        group: "torch.distributed.ProcessGroup|None" = None,
        logits_scale_factor: float = 1.0,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        split_index: int = 0,
        num_splits: int = 1,
        sequence_parallel_logits: bool = False,
        kwargs: dict[str, typing.Any],
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.entropy_loss import entropy_loss_forward_backward

        if prediction_distance > 0:
            raise NotImplementedError()

        reference_model_logits = kwargs[f"{self.reference_model}_logits"].flatten(0, -2)

        # Get the local chunk.
        if sequence_parallel_logits:
            from fast_llm.core.ops import split_op

            reference_model_logits = split_op(reference_model_logits, group, 0)

        # Get the chunk for the current split.
        if num_splits > 1:
            reference_model_logits = reference_model_logits.chunk(num_splits)[split_index]

        implementation = (
            EntropyLossImplementation.fused
            if self.implementation == EntropyLossImplementation.auto
            else self.implementation
        )
        return entropy_loss_forward_backward(
            logits,
            reference_model_logits,
            loss_mask,
            grad_output=grad_output,
            group=group,
            implementation=implementation,
            logits_scale_factor=logits_scale_factor,
            temperature=self.temperature,
            target_format=TargetFormat.labels,
            entropy_loss_type=self.loss_type,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "dpo"})
class LanguageModelDPOLossConfig(LanguageModelLossConfig):
    """Direct Preference Optimization (DPO) loss for alignment."""

    _name: typing.ClassVar[str] = "DPO_loss"
    _abstract: typing.ClassVar[bool] = False

    beta: float = Field(
        default=1.0,
        hint=FieldHint.core,
        desc="Beta parameter for DPO loss (controls strength of preference optimization).",
        valid=check_field(Assert.gt, 0.0),
    )

    reference_model: str = Field(
        desc="Name of the reference model to use for dpo.",
        hint=FieldHint.feature,
    )

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        *,
        group: "torch.distributed.ProcessGroup|None" = None,
        logits_scale_factor: float = 1.0,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        split_index: int = 0,
        num_splits: int = 1,
        sequence_parallel_logits: bool = False,
        kwargs: dict[str, typing.Any],
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.functional.dpo import compute_dpo_loss

        if num_splits > 1:
            raise NotImplementedError()
        if prediction_distance > 0:
            raise NotImplementedError()

        if logits_scale_factor != 1.0:
            # TODO: Make more efficient.
            logits = logits * logits_scale_factor

        reference_model_logits = kwargs[f"{self.reference_model}_logits"].flatten(0, -2)
        target = kwargs[LanguageModelLossKwargs.labels]

        if sequence_parallel_logits:
            from fast_llm.core.ops import split_op

            reference_model_logits = split_op(reference_model_logits, group, 0)
            target = split_op(target, group, 0)

        chosen_spans = kwargs[LanguageModelLossKwargs.chosen_spans]
        rejected_spans = kwargs[LanguageModelLossKwargs.rejected_spans]

        return compute_dpo_loss(
            logits=logits,
            targets=target,
            reference_model_logits=reference_model_logits,
            chosen_spans=chosen_spans,
            rejected_spans=rejected_spans,
            beta=self.beta,
            grad_output=grad_output,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "z_loss"})
class LanguageModelZLossConfig(LanguageModelLossConfig):
    """Z-loss regularization to prevent overconfidence."""

    _name: typing.ClassVar[str] = "Z_loss"
    _abstract: typing.ClassVar[bool] = False

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        *,
        group: "torch.distributed.ProcessGroup|None" = None,
        logits_scale_factor: float = 1.0,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        split_index: int = 0,
        num_splits: int = 1,
        sequence_parallel_logits: bool = False,
        kwargs: dict[str, typing.Any],
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.layers.common.auxiliary_loss import z_loss_forward_backward

        # TODO: ====== Support loss mask, vocab_parallel ======
        assert loss_mask is None
        assert group is None

        return z_loss_forward_backward(
            logits,
            grad_output,
            loss_mask,
            logits_scale_factor,
        )
