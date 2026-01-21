import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import EntropyLossImplementation, EntropyLossType
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    pass

    from fast_llm.layers.language_model.loss.dpo import LanguageModelDPOLoss
    from fast_llm.layers.language_model.loss.entropy_loss import (
        LanguageModelDistillationLoss,
        LanguageModelLabelEntropyLoss,
    )
    from fast_llm.layers.language_model.loss.loss import LanguageModelLoss
    from fast_llm.layers.language_model.loss.z_loss import LanguageModelZLoss


class LanguageModelLossKwargs(BlockKwargs):
    labels = "labels"
    chosen_spans = "chosen_spans"
    rejected_spans = "rejected_spans"
    advantages = "advantages"
    old_log_probabilities = "old_log_probabilities"


@config_class(registry=True)
class LanguageModelLossConfig(Config):
    _abstract: typing.ClassVar[bool] = True

    weight: float = Field(
        default=1.0,
        hint=FieldHint.core,
        desc="Weight for this loss in the total loss computation.",
        valid=check_field(Assert.geq, 0.0),
    )

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        name: str,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        vocab_parallel: bool = False,
        num_splits: int = 1,
        logits_scale_factor: float = 1.0,
        weight: float = 1.0,
    ):
        return self.loss_class(
            self,
            distributed_config,
            name=name,
            prediction_distance=prediction_distance,
            prediction_heads=prediction_heads,
            vocab_parallel=vocab_parallel,
            num_splits=num_splits,
            logits_scale_factor=logits_scale_factor,
            weight=weight,
        )

    @property
    def loss_class(self) -> "type[LanguageModelLoss]":
        raise NotImplementedError()

    def get_reference_models(self) -> set[str]:
        return set()


@config_class(dynamic_type={LanguageModelLossConfig: "label"})
class LanguageModelLabelEntropyLossConfig(LanguageModelLossConfig):
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

    @property
    def loss_class(self) -> "type[LanguageModelLabelEntropyLoss]":
        from fast_llm.layers.language_model.loss.entropy_loss import LanguageModelLabelEntropyLoss

        return LanguageModelLabelEntropyLoss


@config_class(dynamic_type={LanguageModelLossConfig: "distillation"})
class LanguageModelDistillationLossConfig(LanguageModelLossConfig):
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
        default="teacher",
        desc="Name of the reference model for knowledge distillation.",
        hint=FieldHint.feature,
    )
    temperature: float = Field(
        default=1.0,
        hint=FieldHint.optional,
        desc="Temperature for teacher softmax.",
        valid=check_field(Assert.gt, 0.0),
    )

    @property
    def loss_class(self) -> "type[LanguageModelDistillationLoss]":
        from fast_llm.layers.language_model.loss.entropy_loss import LanguageModelDistillationLoss

        return LanguageModelDistillationLoss

    def get_reference_models(self) -> set[str]:
        return {self.reference_model}


@config_class(dynamic_type={LanguageModelLossConfig: "dpo"})
class LanguageModelDPOLossConfig(LanguageModelLossConfig):
    """Direct Preference Optimization (DPO) loss for alignment."""

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

    @property
    def loss_class(self) -> "type[LanguageModelDPOLoss]":
        from fast_llm.layers.language_model.loss.dpo import LanguageModelDPOLoss

        return LanguageModelDPOLoss

    def get_reference_models(self) -> set[str]:
        return {self.reference_model}


@config_class(dynamic_type={LanguageModelLossConfig: "z_loss"})
class LanguageModelZLossConfig(LanguageModelLossConfig):
    """Z-loss regularization to prevent overconfidence."""

    _abstract: typing.ClassVar[bool] = False

    @property
    def loss_class(self) -> "type[LanguageModelZLoss]":
        from fast_llm.layers.language_model.loss.z_loss import LanguageModelZLoss

        return LanguageModelZLoss
