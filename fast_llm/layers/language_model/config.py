import functools

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.config_utils.initialization import InitializationConfig, Initializer, init_normal_
from fast_llm.functional.config import CrossEntropyImpl, DistillationLossImpl
from fast_llm.layers.block.config import BlockConfig, BlockKwargs
from fast_llm.utils import Assert


class LanguageModelLossNames:
    language_model_loss = "language_model_loss"
    z_loss = "z_loss"
    dpo_loss = "dpo_loss"
    distil_lm_loss = "distillation_language_model_loss"  # the next token perdiciton of combined distillation loss
    distillation_loss = "distillation_loss"

    @staticmethod
    def multi_token_prediction_loss(index: int) -> str:
        if index == 0:
            return LanguageModelLossNames.language_model_loss
        return f"language_model_loss_{index}"


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
class LanguageModelBaseConfig(BaseModelConfig):
    # TODO: block
    transformer: BlockConfig = Field(
        desc="Configuration for the transformer architecture.",
        hint=FieldHint.architecture,
    )
    vocab_size: int = Field(
        default=49152,
        desc="Size of the vocabulary, i.e., number of vocabulary embeddings and logits.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    absolute_position_embeddings: int | None = Field(
        # TODO: backward compatibility?
        default=None,
        desc="Number of absolute position embeddings, if applicable.",
        hint=FieldHint.architecture,
    )
    tie_word_embeddings: bool = Field(
        default=True,
        desc="Tie the output weights (logits) with the vocabulary embedding.",
        hint=FieldHint.architecture,
    )
    prediction_heads: int = Field(
        default=1,
        desc="Number of multi-token prediction heads.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    enable_dpo: bool | None = Field(
        default=False,
        desc="Whether to enable DPO loss",
        hint=FieldHint.feature,
    )
    dpo_beta: float | None = Field(
        default=1.0,
        desc="Beta value for DPO loss.",
        hint=FieldHint.feature,
    )
    dpo_reference_model: str | None = Field(
        default=None,
        desc="Name of the reference model to use for dpo.",
        hint=FieldHint.feature,
    )
    cross_entropy_impl: CrossEntropyImpl = Field(
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
    distillation_model: str | None = Field(
        default=None,
        desc="Name of the reference model to use for knowledge distillation."
        "If provided, replace the loss with a distillation loss.",
        hint=FieldHint.feature,
    )
    # Tensor-parallel word embeddings
    # (Default init std is different, dropout won't match, needs seq_first = False.)
    # (disable to allow for sequence-parallel embeddings and logits, better for larger models)
    parallel_embeddings: bool = Field(
        default=True,
        desc="Allow for tensor-parallel vocabulary embeddings and output weights.",
        doc="Disable to allow for sequence-tensor-parallel input tokens, logits and cross-entropy computation."
        " The sequence-tensor-parallel version typically runs faster, but may incur a small memory cost."
        " Affects RNG for initialization and dropout.",
        hint=FieldHint.performance,
    )
    sequence_first: bool | None = Field(
        default=None,
        desc="Override the default dimension ordering",
        doc="By default, the hidden states are stored with dimensions (batch, sequence, ...), as it makes attention more efficient."
        " However, some settings such as sequence-tensor/data/pipelineo-parallel instead require the ordering (sequence, batch, ...)."
        " Setting this parameter overrides the default choice. Note that setting to `False` will either do nothing or raise an error.",
        hint=FieldHint.testing,
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
    embeddings_lr_scale: float | None = Field(
        default=None,
        desc="Learning rate scale for the word embeddings.",
        doc="May be used to freeze some layers by setting their scale to zero.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )
    output_lr_scale: float | None = Field(
        default=None,
        desc="Custom learning rate scale for the output weights.",
        doc="May be used to freeze the output weights by setting their scale to zero.",
        hint=FieldHint.feature,
    )
    prediction_loss_coefficient: list[float] | None = Field(
        default=None,
        desc="Loss coefficient for each prediction head.",
        doc="If not provided, all heads are equally weighted.",
        hint=FieldHint.feature,
    )
    word_embedding_weight_initialization: InitializationConfig = Field(
        desc="Initialization configuration for word embeddings. Default: normal(std=hidden_size**-0.5)",
        hint=FieldHint.feature,
    )
    position_embedding_weight_initialization: InitializationConfig = Field(
        desc="Initialization configuration for position embeddings. Default: normal(hidden_size**-0.5)",
        hint=FieldHint.feature,
    )
    output_weight_initialization: InitializationConfig = Field(
        desc="Initialization configuration for untied output weights. Default: normal(hidden_size**-0.5)",
        hint=FieldHint.feature,
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.language_model_loss_factor is None:
                if self.distillation_model is None:
                    self.language_model_loss_factor = 1.0
                else:
                    self.language_model_loss_factor = 0.0
        super()._validate()
        if self.distillation_model is not None:
            if self.prediction_heads > 1:
                raise NotImplementedError("Multi-token prediction not supported with distillation.")
        if isinstance(self.prediction_loss_coefficient, list):
            Assert.eq(len(self.prediction_loss_coefficient), self.prediction_heads)
            for coeff in self.prediction_loss_coefficient:
                Assert.geq(coeff, 0)
        if self.transformer.per_layer_lr_scale is not None:
            # -1 because the first prediction head's transformer layer is accounted for in num_layers
            # +1 because the layer index starts at 1
            Assert.eq(
                len(self.transformer.per_layer_lr_scale), self.transformer.num_blocks + self.prediction_heads - 1 + 1
            )
        if self.output_weight_initialization.has_initialization:
            assert self.use_absolute_position_embeddings
        if self.output_weight_initialization.has_initialization:
            assert not self.tie_word_embeddings

    @property
    def use_absolute_position_embeddings(self) -> int:
        return self.absolute_position_embeddings is not None

    @functools.cached_property
    def word_embedding_weight_initialization_method(self) -> Initializer:
        if self.word_embedding_weight_initialization.has_initialization:
            return self.word_embedding_weight_initialization.get_initializer()
        else:
            return init_normal_(self.transformer.hidden_size**-0.5)

    @functools.cached_property
    def position_embedding_weight_initialization_method(self) -> Initializer:
        if self.position_embedding_weight_initialization.has_initialization:
            return self.position_embedding_weight_initialization.get_initializer()
        else:
            return init_normal_(self.transformer.hidden_size**-0.5)

    @functools.cached_property
    def output_weight_initialization_method(self) -> Initializer:
        if self.output_weight_initialization.has_initialization:
            return self.output_weight_initialization.get_initializer()
        else:
            return init_normal_(self.transformer.hidden_size**-0.5)
