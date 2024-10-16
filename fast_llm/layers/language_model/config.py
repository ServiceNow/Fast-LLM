from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig, BaseModelConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.functional.config import CrossEntropyImpl
from fast_llm.layers.transformer.config import TransformerArchitectureConfig, TransformerConfig
from fast_llm.utils import Assert


class LanguageModelDimNames:
    # Embedding dimensions
    position_embed = "position_embed"
    vocab = "vocab"
    vocab_tp = "vocab_tp"
    # Misc
    scalar = "scalar"


class LanguageModelLossNames:
    language_model_loss = "language_model_loss"
    z_loss = "z_loss"


class LanguageModelKwargs:
    position_ids = "position_ids"
    # TODO: These are generic
    labels = "labels"
    phase = "phase"


@config_class()
class LanguageModelArchitectureConfig(BaseModelArchitectureConfig):
    transformer: TransformerArchitectureConfig = Field(
        default_factory=TransformerArchitectureConfig,
        desc="Configuration for the transformer architecture.",
        hint=FieldHint.core,
    )
    max_position_embeddings: int = Field(
        default=2048,
        desc="Number of absolute position embeddings, if applicable.",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )
    vocab_size: int = Field(
        default=49152,
        desc="Size of the vocabulary, i.e., number of vocabulary embeddings and logits.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    use_position_embeddings: bool = Field(
        default=None,
        desc="Enable absolute position embeddings. Default: Enable unless using rotary embeddings.",
        hint=FieldHint.feature,
    )
    tie_word_embeddings: bool = Field(
        default=True, desc="Tie the output weights (logits) with the vocabulary embedding.", hint=FieldHint.core
    )

    def _validate(self):
        if self.use_position_embeddings is None:
            self.use_position_embeddings = not self.transformer.use_rotary_embeddings
        super()._validate()

    def setup_tensor_space(self, tensor_space: TensorSpace):
        self.transformer.setup_tensor_space(tensor_space)
        tensor = tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        # Embedding dimensions
        tensor_space.add_tensor_dim(TensorDim(LanguageModelDimNames.position_embed, self.max_position_embeddings))
        # TODO: Need both?
        tensor_space.add_tensor_dim(TensorDim(LanguageModelDimNames.vocab, self.vocab_size))
        tensor_space.add_tensor_dim(TensorDim(LanguageModelDimNames.vocab_tp, self.vocab_size, tensor))

    @property
    def num_absolute_position_embeddings(self):
        # TODO: Rename from max embeddings.
        return self.max_position_embeddings if self.use_absolute_position_embeddings else None

    @property
    def use_absolute_position_embeddings(self):
        # TODO: Set through num embeddings instead instead.
        return self.use_position_embeddings

    @classmethod
    def from_flat_dict(
        cls,
        default: dict[str],
        strict: bool = True,
    ):
        # The backward compatibility fix in `NormalizationArchitectureConfig`
        # won't work for older checkpoints saved with a flat config.
        # TODO v0.2: Remove flat format
        cls._handle_renamed_field(default, "normalization_type", "type")
        cls._handle_renamed_field(default, "layer_norm_eps", "epsilon")
        cls._handle_renamed_field(default, "zero_centered_normalization", "zero_centered")
        return super().from_flat_dict(default, strict)


@config_class()
class LanguageModelBaseConfig(LanguageModelArchitectureConfig, BaseModelConfig):
    """
    A configuration class for defining the full model configuration of a transformer model.
    The parameters of this class are assumed not to be essential to the model architecture, i.e.,
    it makes sense to re-load a model with a different config, (ex. for fine-tuning)
    as long as the `TransformerConfigBase` is the same.
    Currently hard-coded to a GPT-style model.
    TODO: Generalize model configurations.
    """

    architecture_cls = LanguageModelArchitectureConfig

    transformer: TransformerConfig = Field(
        default_factory=TransformerConfig, desc="Configuration for the transformer.", hint=FieldHint.core
    )
    init_method_std_embed: float = Field(
        default=None,
        desc="Initialization scale for the vocabulary embedding and output weights (logits).",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )
    cross_entropy_impl: CrossEntropyImpl = Field(
        default=CrossEntropyImpl.auto,
        desc="Implementation for the cross-entropy computation.",
        hint=FieldHint.performance,
    )
    cross_entropy_splits: int | None = Field(
        default=None,
        desc="Split the logit and cross-entropy computation into this many fragment, to reduce memory usage.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
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
    logits_scale_factor: float = Field(
        default=1.0,
        desc="Multiply output logits by scale factor.",
        doc="Useful in muP setting, since we need to adjust the output logits by the width factor."
        " Since we are mupltiplying the output logits, under muP the scale factor should be < 1.0.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )

    def _validate(self):
        if self.transformer.init_method_std is None:
            self.transformer.init_method_std = self.transformer.hidden_size**-0.5
        if self.init_method_std_embed is None:
            self.init_method_std_embed = self.transformer.init_method_std
        super()._validate()
