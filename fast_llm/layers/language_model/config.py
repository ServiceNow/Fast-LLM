import abc
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.parameter import OptionalParameterConfig, ParameterConfig, combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.config import BlockConfig, BlockSequenceConfig
from fast_llm.layers.common.normalization.config import NormalizationConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.language_model.loss.config import (
    LanguageModelLabelEntropyLossConfig,
    LanguageModelLossConfig,
    LanguageModelLossKwargs,
)
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
    from fast_llm.layers.language_model.head import LanguageModelHead, LanguageModelHeadBase
    from fast_llm.layers.language_model.language_model import LanguageModel
    from fast_llm.layers.language_model.multi_token_prediction import MultiTokenPrediction


class LanguageModelKwargs(LanguageModelLossKwargs):
    token_ids = "token_ids"
    position_ids = "position_ids"
    token_map = "token_map"
    sample_map = "sample_map"
    embedding_map = "embedding_map"
    # TODO: These are generic
    phase = "phase"
    loss_mask = "loss_mask"
    mask_inputs = "mask_inputs"


LM_HEAD_LOSS_NAME = "lm_head_loss"


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
    # TODO: Option to chose whether to split in batch or sequence dimension?
    #   (Currently split merged batch and sequence, depends on `sequence_first`)
    cross_entropy_splits: int = Field(
        default=1,
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

    def _validate(self) -> None:
        with self._set_implicit_default():
            if not self.losses:
                if "losses" not in self._explicit_fields:
                    self.losses = {"lm_loss": LanguageModelLabelEntropyLossConfig()}
        super()._validate()
        assert LM_HEAD_LOSS_NAME not in self.losses

    @property
    def max_prediction_distance(self) -> int:
        return 1


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
