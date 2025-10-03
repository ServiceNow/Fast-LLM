import abc
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import ModuleConfig
from fast_llm.engine.config_utils.parameter import OptionalParameterConfig, ParameterConfig, combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.functional.config import CrossEntropyImpl, DistillationLossImpl
from fast_llm.layers.block.config import BlockConfig, BlockKwargs, BlockSequenceConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
    from fast_llm.layers.language_model.head import LanguageModelHead, LanguageModelHeadBase
    from fast_llm.layers.language_model.multi_token_prediction import MultiTokenPrediction


class LanguageModelKwargs(BlockKwargs):
    position_ids = "position_ids"
    # TODO: These are generic
    labels = "labels"
    phase = "phase"
    chosen_spans = "chosen_spans"
    rejected_spans = "rejected_spans"
    loss_mask = "loss_mask"
    mask_inputs = "mask_inputs"


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
    hidden_size: int = Field(
        default=1024,
        desc="Size of the model's main hidden dimension, e.g., for its input and output layers.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
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
    # TODO: Cleanup
    output_weight: ParameterConfig = Field(
        desc="Configuration for the LM output layer (weight). Ignored for tied embeddings",
        hint=FieldHint.architecture,
    )
    cross_entropy_implementation: CrossEntropyImpl = Field(
        default=CrossEntropyImpl.auto,
        desc="Implementation for the cross-entropy computation.",
        hint=FieldHint.performance,
    )
    distillation_loss_implementation: DistillationLossImpl = Field(
        default=DistillationLossImpl.cross_entropy,
        desc="Implementation for the distillation cross-entropy computation.",
        hint=FieldHint.performance,
    )
    cross_entropy_splits: int | None = Field(
        default=None,
        desc="Split the logit and cross-entropy computation into this many fragment, to reduce memory usage.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
    )
    logit_z_loss: float = Field(
        default=0.0,
        desc="Regularize the logits with Z-loss.",
        doc="We recommend 1e-4 for stability, as used for training PaLM.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    language_model_loss_factor: float = Field(
        default=None,
        desc="Factor to scale the language modeling loss by when using distillation.",
        hint=FieldHint.feature,
    )
    distillation_loss_factor: float = Field(
        default=1.0,
        desc="Factor to scale the distillation loss by when using distillation.",
        hint=FieldHint.feature,
    )
    logits_scale_factor: float = Field(
        default=1.0,
        desc="Multiply output logits by scale factor.",
        doc="Useful in muP setting, since we need to adjust the output logits by the width factor."
        " Since we are mupltiplying the output logits, under muP the scale factor should be < 1.0.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    teacher_softmax_temperature: float = Field(
        default=1.0,
        desc="Divides distillation target logits by this factor.",
        doc="Divides distillation target logits by this factor.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    dpo_reference_model: str | None = Field(
        default=None,
        desc="Name of the reference model to use for dpo.",
        hint=FieldHint.feature,
    )
    dpo_beta: float | None = Field(
        default=1.0,
        desc="Beta value for DPO loss.",
        hint=FieldHint.feature,
    )
    distillation_model: str | None = Field(
        default=None,
        desc="Name of the reference model to use for knowledge distillation."
        "If provided, replace the loss with a distillation loss.",
        hint=FieldHint.feature,
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

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.language_model_loss_factor is None:
                if self.distillation_model is None:
                    self.language_model_loss_factor = 1.0
                else:
                    self.language_model_loss_factor = 0.0
        super()._validate()
        assert self.dpo_reference_model is None or self.distillation_model is None  # currently don't support both

    @property
    def max_prediction_distance(self) -> int:
        return 1

    @property
    def enable_dpo(self) -> bool:
        return self.dpo_reference_model is not None


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
class LanguageModelConfig(ModuleConfig):
    # TODO: block
    decoder: BlockSequenceConfig = Field(
        desc="Configuration for the language model decoder.",
        hint=FieldHint.architecture,
    )
    embeddings: LanguageModelEmbeddingsConfig = Field()
    head: LanguageModelHeadBaseConfig = Field()
    # TODO: Allow overriding in sub-models?
    peft: PeftConfig = Field(
        desc="Configuration for parameter-efficient fine tuning.",
        hint=FieldHint.architecture,
    )
    tied_embedding_weight: bool = Field(
        default=False,
        desc="Tie the output weights (logits) with the vocabulary embedding.",
        hint=FieldHint.architecture,
    )
    sequence_first: bool | None = Field(
        default=None,
        desc="Override the default dimension ordering",
        doc="By default, the hidden states are stored with dimensions (batch, sequence, ...), as it makes attention more efficient."
        " However, some settings such as sequence-tensor/data/pipelineo-parallel instead require the ordering (sequence, batch, ...)."
        " Setting this parameter overrides the default choice. Note that setting to `False` will either do nothing or raise an error.",
        hint=FieldHint.testing,
    )
