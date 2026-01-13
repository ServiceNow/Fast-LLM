import abc
import typing
import warnings
from functools import cached_property

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.parameter import OptionalParameterConfig, ParameterConfig, combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import EntropyLossImplementation, TargetFormat, TritonConfig
from fast_llm.layers.block.config import BlockConfig, BlockKwargs, BlockSequenceConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.core.distributed import ProcessGroup
    from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
    from fast_llm.layers.language_model.head import LanguageModelHead, LanguageModelHeadBase
    from fast_llm.layers.language_model.language_model import LanguageModel
    from fast_llm.layers.language_model.multi_token_prediction import MultiTokenPrediction


class TargetsKwargs:
    lm_target = "preprocessed_lm_target"
    dpo_target = "preprocessed_dpo_target"
    reference_model_logits = "reference_model_logits"
    dpo_reference_model_logits = "dpo_reference_model_logits"


class LanguageModelKwargs(BlockKwargs):
    token_ids = "token_ids"
    position_ids = "position_ids"
    token_map = "token_map"
    sample_map = "sample_map"
    embedding_map = "embedding_map"
    # TODO: These are generic
    labels = "labels"
    phase = "phase"
    chosen_spans = "chosen_spans"
    rejected_spans = "rejected_spans"
    loss_mask = "loss_mask"
    mask_inputs = "mask_inputs"


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
        group: "ProcessGroup|None" = None,
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
        Returns loss name for logging as '<registered_loss_name>(<self._name>)',
        e.g. lm_loss(CE_loss), distillation(FwdKL_loss)
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
        group: "ProcessGroup|None" = None,
    ) -> dict[str, "torch.Tensor"]:
        pass


@config_class(dynamic_type={LanguageModelLossConfig: "cross_entropy"})
class CrossEntropyLanguageModelLossConfig(LanguageModelLossConfig):
    _name: typing.ClassVar[str] = "CE_loss"
    _abstract: typing.ClassVar[bool] = False

    implementation: EntropyLossImplementation = Field(
        default=EntropyLossImplementation.auto,
        desc="Implementation for the cross-entropy computation.",
        hint=FieldHint.performance,
    )

    temperature: float = Field(
        default=1.0,
        hint=FieldHint.optional,
        desc="Temperature for teacher softmax.",
        valid=check_field(Assert.gt, 0.0),
    )

    def get_targets(
        self,
        kwargs: dict | None = None,
        prediction_distance: int | None = None,
        prediction_heads: int | None = None,
        sequence_parallel_logits: bool | None = None,
        group: "ProcessGroup|None" = None,
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
        from fast_llm.functional.cross_entropy import entropy_loss_forward_backward

        target = kwargs.get(TargetsKwargs.lm_target)
        implementation = self.implementation
        if implementation == EntropyLossImplementation.auto:
            if vocab_parallel:
                implementation = EntropyLossImplementation.fused
            elif TritonConfig.TRITON_ENABLED:
                implementation = EntropyLossImplementation.triton
            else:
                implementation = EntropyLossImplementation.fused

        return entropy_loss_forward_backward(
            logits=logits.flatten(0, -2),
            target=target,
            loss_mask=None,  # Labels are already masked
            grad_output=grad_output,
            group=group,
            implementation=implementation,
            logits_scale_factor=logits_scale_factor,
            teacher_softmax_temperature=self.temperature,
            target_format=TargetFormat.labels,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "forward_kl_distillation"})
class ForwardKLDistillationLossConfig(LanguageModelLossConfig):
    """Forward KL divergence KL(p||q) for distillation (mode-covering)."""

    _name: typing.ClassVar[str] = "FwdKL_loss"
    _abstract: typing.ClassVar[bool] = False

    temperature: float = Field(
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
        group: "ProcessGroup|None" = None,
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
        group: "ProcessGroup|None" = None,
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
            teacher_softmax_temperature=self.temperature,
            target_format=TargetFormat.logits,
        )


@config_class(dynamic_type={LanguageModelLossConfig: "reverse_kl_distillation"})
class ReverseKLLossConfig(ForwardKLDistillationLossConfig):
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
        group: "ProcessGroup|None" = None,
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
            teacher_softmax_temperature=self.temperature,
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
        group: "ProcessGroup|None" = None,
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
        group: "ProcessGroup|None" = None,
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


@config_class(dynamic_type={LanguageModelLossConfig: "z_loss"})
class ZLossConfig(LanguageModelLossConfig):
    """Z-loss regularization to prevent overconfidence."""

    _name: typing.ClassVar[str] = "Z_loss"
    _abstract: typing.ClassVar[bool] = False

    def get_targets(
        self,
        kwargs: dict | None = None,
        prediction_distance: int | None = None,
        prediction_heads: int | None = None,
        sequence_parallel_logits: bool | None = None,
        group: "ProcessGroup|None" = None,
    ) -> dict[str, "torch.Tensor"]:
        return {}

    def get_loss(
        self,
        logits: "torch.Tensor",
        loss_mask: "torch.Tensor | None",
        grad_output: float | None = None,
        group: "ProcessGroup|None" = None,
        logits_scale_factor: float | None = None,
        vocab_parallel: bool = False,
        kwargs: dict | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        from fast_llm.layers.common.auxiliary_loss import z_loss_forward_backward

        # TODO: ====== Support loss mask, vocab_parallel ======
        assert loss_mask is None
        assert group is None

        return z_loss_forward_backward(
            logits=logits.flatten(0, -2),
            grad_output=grad_output,
            logits_scale_factor=logits_scale_factor,
        )


@config_class()
class LanguageModelEmbeddingsConfig(BlockConfig):
    _abstract = False
    word_embeddings: ParameterConfig = Field(
        desc="Configuration for the word embedding (weight).",
        hint=FieldHint.architecture,
    )
    position_embeddings: OptionalParameterConfig = Field(
        desc="Configuration for the word embedding (weight).",
        hint=FieldHint.architecture,
    )
    vocab_size: int = Field(
        default=49152,
        desc="Size of the vocabulary, i.e., number of vocabulary embeddings and logits.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    num_position_embeddings: int = Field(
        default=2048,
        desc="Number of absolute position embeddings, if applicable.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    cross_document_position_embeddings: bool = Field(
        default=True,
        desc="Allow for cross-document position embeddings.",
        doc="Disable to reset position ids at the beginning of each document.",
        hint=FieldHint.feature,
    )

    dropout: float = Field(
        default=0.0,
        desc="Dropout applied to the embedding layer.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    full_precision_residual: bool = Field(
        default=False,
        desc="Store the residuals for the model in full precision (`optimization_dtype`).",
        hint=FieldHint.stability,
    )

    # Tensor-parallel word embeddings
    # (Default init std is different, dropout won't match, needs seq_first = False.)
    # (disable to allow for sequence-parallel embeddings and logits, better for larger models)
    vocab_parallel: bool = Field(
        default=True,
        desc="Allow for tensor-parallel vocabulary embeddings and output weights.",
        doc="Disable to allow for sequence-tensor-parallel input tokens, logits and cross-entropy computation."
        " The sequence-tensor-parallel version typically runs faster, but may incur a small memory cost."
        " Affects RNG for initialization and dropout.",
        hint=FieldHint.performance,
    )

    @property
    def layer_class(self) -> "type[LanguageModelEmbedding]":
        from fast_llm.layers.language_model.embedding import LanguageModelEmbedding

        return LanguageModelEmbedding


@config_class(registry=True)
class LanguageModelHeadBaseConfig(BlockConfig):
    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is LanguageModelHeadBaseConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return LanguageModelHeadConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        embeddings_config: LanguageModelEmbeddingsConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "LanguageModelHeadBase":
        return self.layer_class(
            self,
            distributed_config,
            embeddings_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
        )

    @property
    @abc.abstractmethod
    def max_prediction_distance(self) -> int:
        pass


@config_class(dynamic_type={LanguageModelHeadBaseConfig: "language_model_head"})
class LanguageModelHeadConfig(LanguageModelHeadBaseConfig):
    _abstract = False
    normalization: NormalizationConfig = Field(
        desc="Configuration for the final normalization layer.",
        hint=FieldHint.architecture,
    )
    losses: dict[str, LanguageModelLossConfig] = Field(
        default_factory=dict,
        desc="A dictionary of loss names and their configurations.",
        hint=FieldHint.core,
    )
    # TODO: Cleanup
    output_weight: ParameterConfig = Field(
        desc="Configuration for the LM output layer (weight). Ignored for tied embeddings",
        hint=FieldHint.architecture,
    )
    cross_entropy_splits: int | None = Field(
        default=None,
        desc="Split the logit and cross-entropy computation into this many fragment, to reduce memory usage.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    logits_scale_factor: float = Field(
        default=1.0,
        desc="Multiply output logits by scale factor.",
        doc="Useful in muP setting, since we need to adjust the output logits by the width factor."
        " Since we are mupltiplying the output logits, under muP the scale factor should be < 1.0.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        embeddings_config: LanguageModelEmbeddingsConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        loss_coefficient: float = 1.0,
    ):
        return self.layer_class(
            self,
            distributed_config,
            embeddings_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
            prediction_distance=prediction_distance,
            prediction_heads=prediction_heads,
            loss_coefficient=loss_coefficient,
        )

    @property
    def layer_class(self) -> "type[LanguageModelHead]":
        from fast_llm.layers.language_model.head import LanguageModelHead

        return LanguageModelHead

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        removed_fields = ["distillation_loss_factor", "distillation_model", "language_model_loss_factor"]
        for field in removed_fields:
            if field in default:
                warnings.warn(
                    f"Field `{field}` has been removed from {cls.__name__}. "
                    "Loss configuration should now be done via the `losses` field.",
                    DeprecationWarning,
                )
                default.pop(field)
        return super()._from_dict(default, strict=strict)

    def _validate(self) -> None:
        with self._set_implicit_default():
            if not self.losses:
                if "losses" not in self._explicit_fields:
                    self.losses = {"lm_loss": CrossEntropyLanguageModelLossConfig()}
        super()._validate()
        if DPOLossConfig in self._loss_configs:
            assert ForwardKLDistillationLossConfig not in self._loss_configs.keys()  # currently don't support both
            assert ReverseKLLossConfig not in self._loss_configs.keys()  # currently don't support both
        if (
            ForwardKLDistillationLossConfig in self._loss_configs.keys()
            and ReverseKLLossConfig in self._loss_configs.keys()
        ):
            assert (
                self._loss_configs[ForwardKLDistillationLossConfig].distillation_model
                == self._loss_configs[ReverseKLLossConfig].distillation_model
            ), "Distillation losses must use the same teacher."

    @cached_property
    def _loss_configs(self) -> dict[type, LanguageModelLossConfig]:
        return {loss.__class__: loss for loss in self.losses.values()}

    @property
    def max_prediction_distance(self) -> int:
        return 1

    @property
    def enable_dpo(self) -> bool:
        return DPOLossConfig in self._loss_configs.keys()

    @property
    def enable_distillation(self) -> bool:
        return (
            ForwardKLDistillationLossConfig in self._loss_configs.keys()
            or ReverseKLLossConfig in self._loss_configs.keys()
        )

    @property
    def requires_loss_masks(self) -> bool:
        return self.enable_distillation

    @property
    def distillation_model(self) -> str | None:
        for loss_type in [ForwardKLDistillationLossConfig, ReverseKLLossConfig]:
            if loss_type in self._loss_configs:
                return self._loss_configs[loss_type].distillation_model
        return None

    @property
    def dpo_reference_model(self) -> str | None:
        if DPOLossConfig in self._loss_configs:
            return self._loss_configs[DPOLossConfig].dpo_reference_model
        return None


@config_class(dynamic_type={LanguageModelHeadBaseConfig: "multi_token_prediction"})
class MultiTokenPredictionConfig(LanguageModelHeadBaseConfig):
    _abstract = False
    # Needs to be `DecoderBlockConfig` for the `return_input` interface.
    # TODO: Make a generic wrapper for returning input instead?
    block: DecoderBlockConfig = Field(
        desc="Configuration for the decoder block before each head.",
        hint=FieldHint.architecture,
    )
    # TODO: Generalize? (needs the extra initialization arguments)
    head: LanguageModelHeadConfig = Field(
        desc="Configuration for the multi-token-prediction heads.",
        hint=FieldHint.architecture,
    )
    prediction_heads: int = Field(
        default=1,
        desc="Prediction heads.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    prediction_loss_coefficient: list[float] | None = Field(
        default=None,
        desc="Loss coefficient for each prediction head.",
        doc="If not provided, all heads are equally weighted.",
        hint=FieldHint.feature,
    )

    def _validate(self) -> None:
        super()._validate()
        if isinstance(self.prediction_loss_coefficient, list):
            Assert.eq(len(self.prediction_loss_coefficient), self.prediction_heads)
            for coeff in self.prediction_loss_coefficient:
                Assert.geq(coeff, 0)

    @property
    def layer_class(self) -> "type[MultiTokenPrediction]":
        from fast_llm.layers.language_model.multi_token_prediction import MultiTokenPrediction

        return MultiTokenPrediction

    @property
    def max_prediction_distance(self) -> int:
        return self.prediction_heads


@config_class()
class LanguageModelConfig(BlockConfig):
    decoder: BlockSequenceConfig = Field(
        desc="Configuration for the language model decoder.",
        hint=FieldHint.architecture,
    )
    embeddings: LanguageModelEmbeddingsConfig = Field(
        hint=FieldHint.architecture,
        desc="Configuration for the language model embeddings.",
    )
    head: LanguageModelHeadBaseConfig = Field(
        hint=FieldHint.architecture, desc="Configuration for the language model head(s)."
    )
    tied_embedding_weight: bool = Field(
        default=False,
        desc="Tie the output weights (logits) with the vocabulary embedding.",
        hint=FieldHint.architecture,
    )
    hidden_size: int = Field(
        default=1024,
        desc="Size of the model's main hidden dimension, e.g., for its input and output layers.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    sequence_first: bool | None = Field(
        default=None,
        desc="Override the default dimension ordering",
        doc="By default, the hidden states are stored with dimensions (batch, sequence, ...), as it makes attention more efficient."
        " However, some settings such as sequence-tensor/data/pipelineo-parallel instead require the ordering (sequence, batch, ...)."
        " Setting this parameter overrides the default choice. Note that setting to `False` will either do nothing or raise an error.",
        hint=FieldHint.testing,
    )

    @property
    def layer_class(self) -> "type[LanguageModel]":
        from fast_llm.layers.language_model.language_model import LanguageModel

        return LanguageModel
