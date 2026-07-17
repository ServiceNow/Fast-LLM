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
    CombinableLossConfig,
    LanguageModelLabelEntropyLossConfig,
    LanguageModelLossConfig,
    LanguageModelLossKwargs,
    LossImplementation,
    MonolithicLossConfig,
)
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
    from fast_llm.layers.language_model.head import LanguageModelHead
    from fast_llm.layers.language_model.language_model import LanguageModel
    from fast_llm.layers.language_model.multi_token_prediction import MultiTokenPrediction


class LanguageModelKwargs(LanguageModelLossKwargs):
    token_ids = "token_ids"
    position_ids = "position_ids"
    token_map = "token_map"
    sample_map = "sample_map"
    embedding_map = "embedding_map"
    num_documents_in_batch = "num_documents_in_batch"
    # Cumulative document count at the start of the step.
    documents_seen = "documents_seen"
    # TODO: These are generic
    phase = "phase"
    loss_mask = "loss_mask"


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
    embedding_scale: float = Field(
        default=1.0,
        desc="Multiplicative scale applied to word embeddings after lookup.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )

    @property
    def layer_class(self) -> "type[LanguageModelEmbedding]":
        from fast_llm.layers.language_model.embedding import LanguageModelEmbedding

        return LanguageModelEmbedding


@config_class()
class LanguageModelHeadConfig(BlockConfig):
    _abstract = False
    normalization: NormalizationConfig = Field(
        desc="Configuration for the final normalization layer.",
        hint=FieldHint.architecture,
    )
    losses: dict[str, LanguageModelLossConfig] = Field(
        default_factory=dict,
        desc="A dictionary of loss names and their configurations. "
        "If not specified, a cross-entropy loss with respect to the targets will be used.",
        hint=FieldHint.core,
    )
    loss_implementation: LossImplementation = Field(
        default=LossImplementation.auto,
        desc="How to realize the losses. `auto`/`compiled`/`triton` fuse the combinable losses into a single"
        " shared-softmax kernel (`auto` picks triton when available and eligible, else compiled); `per_loss`"
        " runs each loss on its own softmax.",
        hint=FieldHint.expert,
    )
    # TODO: Cleanup
    output_weight: ParameterConfig = Field(
        desc="Configuration for the LM output layer (weight). Ignored for tied embeddings",
        hint=FieldHint.architecture,
    )
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
    final_logit_softcap: float | None = Field(
        default=None,
        desc="Soft-cap applied to logits before loss: logits = tanh(logits / cap) * cap.",
        hint=FieldHint.architecture,
        valid=skip_valid_if_none(check_field(Assert.gt, 0)),
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

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        embeddings_config: LanguageModelEmbeddingsConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        block_config: DecoderBlockConfig | None = None,
    ) -> "tuple[LanguageModelHead, MultiTokenPrediction]":
        from fast_llm.layers.language_model.head import LanguageModelHead
        from fast_llm.layers.language_model.multi_token_prediction import MultiTokenPrediction

        return LanguageModelHead(
            self,
            distributed_config,
            embeddings_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
        ), MultiTokenPrediction(
            self,
            distributed_config,
            embeddings_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
            block_config=block_config,
        )

    def _validate(self) -> None:
        super()._validate()
        assert LM_HEAD_LOSS_NAME not in self.losses
        # Surface fusion/grouping errors (e.g. an ineligible `triton` set) at config time, incl. `--validate`.
        self.get_effective_losses()

    def get_effective_losses(self) -> dict[str, LanguageModelLossConfig]:
        # The top-level losses the head builds. Combinable losses are fused into a shared-softmax
        # `MonolithicLoss` unless `loss_implementation` is `per_loss`; a single softmax serves one effective
        # scale, so they are grouped by `logits_scale_factor` (the common head scale applies to all). Each
        # group takes the slot of its first member; non-combinable losses (e.g. DPO) stay standalone.
        losses = self.losses or {"cross_entropy": LanguageModelLabelEntropyLossConfig()}
        if self.loss_implementation == LossImplementation.per_loss:
            return dict(losses)
        use_triton = {
            LossImplementation.auto: None,
            LossImplementation.compiled: False,
            LossImplementation.triton: True,
        }[self.loss_implementation]
        scale_groups: dict[float, dict[str, LanguageModelLossConfig]] = {}
        slots: list[float | tuple[str, LanguageModelLossConfig]] = []
        for name, loss in losses.items():
            if isinstance(loss, CombinableLossConfig):
                if loss.logits_scale_factor not in scale_groups:
                    scale_groups[loss.logits_scale_factor] = {}
                    slots.append(loss.logits_scale_factor)
                scale_groups[loss.logits_scale_factor][name] = loss
            else:
                slots.append((name, loss))
        named = len(scale_groups) > 1
        effective, group_index = {}, 0
        for slot in slots:
            if isinstance(slot, tuple):
                effective[slot[0]] = slot[1]
            else:
                name = f"monolithic_{group_index}" if named else "monolithic"
                effective[name] = MonolithicLossConfig(losses=scale_groups[slot], use_triton=use_triton)
                group_index += 1
        return effective

    def get_reference_models(self) -> set[str]:
        return {reference_model for loss in self.losses.values() for reference_model in loss.get_reference_models()}


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
    head: LanguageModelHeadConfig = Field(
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

    @property
    def layer_class(self) -> "type[LanguageModel]":
        from fast_llm.layers.language_model.language_model import LanguageModel

        return LanguageModel

    def get_reference_models(self) -> set[str]:
        return self.decoder.get_reference_models() | self.head.get_reference_models()
