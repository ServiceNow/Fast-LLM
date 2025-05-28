import enum
import logging
import math
import typing

from blocks import LlambaBlock, LlambaOneBlock

from fast_llm.config import Config, Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointHandler
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.common.config import LLMDimNames
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.layers.transformer.transformer import BaseBlock, TransformerLayer
from fast_llm.models.gpt.config import GPTBatchConfig, PretrainedGPTModelConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.models.gpt.model import GPTInferenceRunner
    from fast_llm.models.hybrid.huggingface import HuggingfaceHybridModelForCausalLM
    from fast_llm.models.hybrid.model import HybridModel
    from fast_llm.models.hybrid.trainer import SSMTrainer

logger = logging.getLogger(__name__)


@config_class(registry=True)
class HybridBlockConfig(Config):
    _abstract = True
    block_class: typing.ClassVar[type[BaseBlock]]
    # config: TransformerConfig | SSMConfig

    lr_scale: list[float] | None = Field(
        default=None,
        desc="Custom learning rate scale for each layer.",
        doc="May be used to freeze some layers by setting their scale to zero.",
        hint=FieldHint.feature,
    )
    hidden_size: int = Field(
        default=1024,
        desc="Hidden size of the block.",
        hint=FieldHint.architecture,
    )


@config_class(dynamic_type={HybridBlockConfig: "transformer"})
class TransformerBlockConfig(HybridBlockConfig, TransformerConfig):
    _abstract = False
    block_class: typing.ClassVar[type[BaseBlock]] = TransformerLayer

    def setup_tensor_space(self, tensor_space: "TensorSpace", block_name: str) -> None:
        TransformerConfig.setup_tensor_space(self, tensor_space, block_name)


@config_class(dynamic_type={HybridBlockConfig: "discrete_mamba2"})
class DiscreteMamba2BlockConfig(HybridBlockConfig, SSMConfig):
    _abstract = False
    block_class: typing.ClassVar[type[BaseBlock]] = LlambaBlock

    # def _validate(self):
    #     self.config.validate()

    def setup_tensor_space(self, tensor_space: TensorSpace, block_name: str) -> None:

        d_inner = int(self.expansion_factor * self.hidden_size) if self.d_inner is None else self.d_inner
        # Hidden dimension
        tensor_space.add_tensor_dim(TensorDim(f"{LLMDimNames.hidden}_{block_name}", self.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.model_dim}_{block_name}", self.hidden_size))
        # Mamba-specific dimensions
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.inner_dim}_{block_name}", d_inner))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.state_dim}_{block_name}", self.state_size))
        tensor_space.add_tensor_dim(
            TensorDim(f"{SSMDimNames.conv_kernel_size}_{block_name}", self.conv_kernel_dimension)
        )

        # as per https://github.com/cartesia-ai/edge/blob/a0e121ebed3d2324c6d762b0e211a08d62583681/cartesia-pytorch/cartesia_pytorch/Llamba/mixers/discrete_mamba2.py#L66C3-L66C4
        headdim = d_inner // self.n_v_heads
        Assert.eq(self.n_v_heads, d_inner // headdim)
        Assert.eq(d_inner % headdim, 0)
        Assert.eq(self.n_v_heads % self.n_qk_heads, 0)

        conv_dim = d_inner + 2 * self.n_qk_heads * self.state_size
        inner_proj_dim = 2 * d_inner + 2 * self.n_qk_heads * self.state_size + self.n_v_heads

        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.qk_heads}_{block_name}", self.n_qk_heads))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.v_heads}_{block_name}", self.n_v_heads))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.inner_proj_mamba2}_{block_name}", inner_proj_dim))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.conv_dim}_{block_name}", conv_dim))

        SSMConfig.setup_tensor_space(self, tensor_space, block_name)


@config_class(dynamic_type={HybridBlockConfig: "mamba"})
class MambaBlockConfig(HybridBlockConfig, SSMConfig):
    _abstract = False
    block_class: typing.ClassVar[type[BaseBlock]] = LlambaOneBlock

    def setup_tensor_space(self, tensor_space: TensorSpace, block_name: str) -> None:

        if self.dt_rank is None:
            mamba_dt_rank = math.ceil(self.hidden_size / 16)
        else:
            mamba_dt_rank = self.dt_rank

        d_inner = int(self.expansion_factor * self.hidden_size) if self.d_inner is None else self.d_inner
        # Hidden dimension
        tensor_space.add_tensor_dim(TensorDim(f"{LLMDimNames.hidden}_{block_name}", self.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.model_dim}_{block_name}", self.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.inner_dim}_{block_name}", d_inner))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.state_dim}_{block_name}", self.state_size))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.dt_rank}_{block_name}", mamba_dt_rank))
        tensor_space.add_tensor_dim(
            TensorDim(f"{SSMDimNames.x_proj_dim}_{block_name}", mamba_dt_rank + self.state_size * 2)
        )
        tensor_space.add_tensor_dim(
            TensorDim(f"{SSMDimNames.conv_kernel_size}_{block_name}", self.conv_kernel_dimension)
        )
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.inner_proj_mamba}_{block_name}", d_inner * 2))

        SSMConfig.setup_tensor_space(self, tensor_space, block_name)


class HybridBlockType(enum.Enum):
    """
    An enum for the available block types, legacy format.
    """

    m = MambaBlockConfig
    m2d = DiscreteMamba2BlockConfig
    t = TransformerBlockConfig


@config_class()
class HybridBaseModelConfig(LanguageModelBaseConfig):
    """
    HybridBaseModelConfig is a configuration class for hybrid models.
    Currently it supports two formats for architecture definition:
     - the old and deprecated format with transformer and ssm fields (t, m2d, m), in wich case all blocks share the same config;
     - and the new format with blocks field, in which case each block can have its own config.
    """

    _abstract = False
    ############################################################################################
    # Note, transformer and ssm are here for legacy reasons
    transformer: TransformerConfig = Field(
        desc="Configuration for the transformer architecture. Note, having transformer and ssm fields in HybridBaseModelConfig is depricated.",
        hint=FieldHint.architecture,
    )

    ssm: SSMConfig = Field(
        desc="Configuration for the SSM architecture. Note, having transformer and ssm fields in HybridBaseModelConfig is depricated.",
        hint=FieldHint.architecture,
    )
    ############################################################################################
    blocks: dict[str, HybridBlockConfig] = Field(
        default=None,
        desc="Named block configurations that can be referenced in block_pattern.",
        hint=FieldHint.architecture,
    )

    hybrid_block_layout: list[str] | None = Field(
        default=None,
        desc=f"Pattern of blocks to use in the model (still supports the previous depricated format with {HybridBlockType.__members__.keys()})",
        hint=FieldHint.core,
    )

    default_mtp_type: str | None = Field(
        default=None,
        desc="Multi-token prediction mixer to use in the model. Can be either one of the blocks, or follow the depricated legacy format: 't' for Transformer, 'm' for Mamba1, 'm2' for discrete Mamba2. If None, will use the last block type in `hybrid_block_layout`.",
        hint=FieldHint.optional,
    )

    # TODO: currently num_layers is defined in TransformerConfig, but ideally this should be migrated to LanguageModelBaseConfig in the future.
    # Hence, for now: the num_layers can be set in the first transformer block, if no transformer blocks used we will fallback to num_layers parameter defined here.
    num_layers: int = Field(
        default=12,
        desc="Number of layers in the transformer.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.geq, 0),
    )

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        """
        Setup the tensor space for the model.
        """
        for block_name, block_config in self.blocks.items():
            block_config.setup_tensor_space(tensor_space, block_name)
        # The first layer's hidden dimension is the input hidden dimension of the model
        tensor_space.add_tensor_dim(
            TensorDim(LLMDimNames.input_hidden, self.blocks[self.hybrid_block_layout[0]].hidden_size)
        )
        # Mark the output hidden dimension of the model
        tensor_space.add_tensor_dim(
            TensorDim(LLMDimNames.output_hidden, self.blocks[self.hybrid_block_layout[-1]].hidden_size)
        )
        super().setup_tensor_space(tensor_space)

    def _validate(self):
        if self.blocks is None:
            logger.warning(
                f"Blocks not set, falling back to old behavior with hybrid_block_layout containing any of {HybridBlockType.__members__.keys()}"
            )
            if self.hybrid_block_layout is None:
                with self._set_implicit_default():
                    logger.warning(
                        f"No hybrid_block_layout found in HybridBaseModelConfig, using default block {HybridBlockType.m2d}"
                    )
                    self.hybrid_block_layout = [HybridBlockType.m2d]

            # Legacy format with t, m, m2d, convert to new format
            Assert.custom(
                lambda _: all(
                    block_type in HybridBlockType.__members__.keys() for block_type in self.hybrid_block_layout
                ),
                f"Invalid block type: {self.hybrid_block_layout}. Must be one of {HybridBlockType.__members__.keys()}",
            )
            blocks = {}
            for block_type in self.hybrid_block_layout:
                if block_type not in blocks:
                    hybrid_block_config_cls = HybridBlockType[block_type].value
                    if hybrid_block_config_cls == TransformerBlockConfig:
                        blocks[block_type] = TransformerBlockConfig.from_dict(self.transformer.to_dict())
                    elif hybrid_block_config_cls == MambaBlockConfig:
                        blocks[block_type] = MambaBlockConfig.from_dict(self.ssm.to_dict())
                    elif hybrid_block_config_cls == DiscreteMamba2BlockConfig:
                        blocks[block_type] = DiscreteMamba2BlockConfig.from_dict(self.ssm.to_dict())
                    else:
                        raise ValueError(f"Invalid block type: {block_type}")
            self.blocks = blocks

        Assert.gt(len(self.hybrid_block_layout), 0)
        # Validate that all pattern entries refer to valid blocks
        for block_name in self.hybrid_block_layout:
            if block_name not in self.blocks:
                raise ValueError(f"Block name '{block_name}' not found in blocks dictionary")

        first_transformer_block_config: TransformerBlockConfig | None = None

        for block_name, block_config in self.blocks.items():
            if isinstance(block_config, TransformerBlockConfig):
                if first_transformer_block_config is None:
                    first_transformer_block_config = block_config
                elif block_config.num_layers != first_transformer_block_config.num_layers:
                    logger.warning(
                        f"Found multiple transformer blocks with different number of layers, using num_layers from the first transformer block for all"
                    )
            block_config.validate()

        # set num_layers from transformer block config if it exists and if num_layers is not set in HybridBaseModelConfig
        # i.e. the resolution hierarchy for num_layers is: HybridBaseModelConfig.num_layers > TransformerBlockConfig.num_layers
        if first_transformer_block_config is not None:
            num_layers = first_transformer_block_config.num_layers
            with self._set_implicit_default():
                if self.num_layers is None:
                    logger.warning(
                        f"TransformerBlockConfig overwrites BaseModelConfig num_layers, setting num_layers = {num_layers}"
                    )
                self.num_layers = num_layers

        # make sure that the hybrid_block_layout length matches the num_layers. If it doesn't, repeat the hybrid_block_layout;
        if len(self.hybrid_block_layout) != self.num_layers:
            if self.transformer.num_layers % len(self.hybrid_block_layout) != 0:
                raise ValueError(
                    f"hybrid_block_layout length {len(self.hybrid_block_layout)} does not match num_layers {self.transformer.num_layers}"
                )
            num_repeats = int(self.transformer.num_layers // len(self.hybrid_block_layout))
            logger.warning(
                f"hybrid_block_layout length {len(self.hybrid_block_layout)} does not match num_layers {self.transformer.num_layers}, will repeat {self.hybrid_block_layout} {num_repeats} times"
            )
            self.hybrid_block_layout = self.hybrid_block_layout * num_repeats

        Assert.eq(len(self.hybrid_block_layout), self.transformer.num_layers)

        with self._set_implicit_default():
            if self.init_method_std_embed is None:
                self.init_method_std_embed = (
                    first_transformer_block_config.init_method_std
                    if first_transformer_block_config is not None
                    else 0.02
                )
            if self.init_method_max_embed is None:
                self.init_method_max_embed = (
                    first_transformer_block_config.init_method_max
                    if first_transformer_block_config is not None
                    else 0.02
                )
            if self.init_method_min_embed is None:
                self.init_method_min_embed = (
                    first_transformer_block_config.init_method_min
                    if first_transformer_block_config is not None
                    else 0.02
                )

        if self.prediction_heads > 1:
            with self._set_implicit_default():
                if self.default_mtp_type is None:
                    logger.warning(
                        f"No default_mtp_type found in HybridBaseModelConfig, using the last block type in hybrid_block_layout: {self.hybrid_block_layout[-1]}"
                    )
                    self.default_mtp_type = self.hybrid_block_layout[-1]
                else:
                    if self.default_mtp_type not in self.hybrid_block_layout:
                        raise ValueError(
                            f"default_mtp_type {self.default_mtp_type} not found in hybrid_block_layout {self.hybrid_block_layout}"
                        )
        super()._validate()


class LLambaHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    name: typing.ClassVar[str] = "llamba"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.hybrid.conversion import LLambaHuggingfaceCheckpointHandler

        return LLambaHuggingfaceCheckpointHandler


class AprielSSMHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    name: typing.ClassVar[str] = "apriel_ssm"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.hybrid.conversion import AprielSSMHuggingfaceCheckpointHandler

        return AprielSSMHuggingfaceCheckpointHandler


class AprielSSMHHybridHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    name: typing.ClassVar[str] = "apriel_ssm_hybrid"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.hybrid.conversion import AprielSSMHHybridHuggingfaceCheckpointHandler

        return AprielSSMHHybridHuggingfaceCheckpointHandler


@config_class()
class HybridModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "hybrid_ssm"
    base_model: HybridBaseModelConfig = FieldUpdate()
    checkpoint_formats = FastLLMModelConfig.checkpoint_formats + (
        LLambaHuggingfaceCheckpointFormat,
        AprielSSMHuggingfaceCheckpointFormat,
        AprielSSMHHybridHuggingfaceCheckpointFormat,
    )

    @classmethod
    def get_model_class(cls) -> type["HybridModel"]:
        from fast_llm.models.hybrid.model import HybridModel

        return HybridModel

    @classmethod
    def get_huggingface_model_for_causal_lm_class(cls) -> type["HuggingfaceHybridModelForCausalLM"]:
        from fast_llm.models.hybrid.huggingface import HuggingfaceHybridModelForCausalLM

        return HuggingfaceHybridModelForCausalLM

    def _validate(self):
        logger.warning(
            "HybridModelConfig is being instantiated. This model is experimental and may not work as expected."
        )
        super()._validate()


@config_class()
class PretrainedHybridModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: HybridModelConfig = FieldUpdate()


@config_class()
class HybridTrainerConfig(PretrainedHybridModelConfig, TrainerConfig):
    data: GPTDataConfig = FieldUpdate()
    batch: GPTBatchConfig = FieldUpdate()
    reference_models: dict[str, PretrainedGPTModelConfig] = FieldUpdate()

    @classmethod
    def get_trainer_class(cls) -> type["SSMTrainer"]:
        from fast_llm.models.hybrid.trainer import SSMTrainer

        return SSMTrainer

    def _validate(self) -> None:
        super()._validate()
        if (name := self.model.base_model.distillation_model) is None:
            Assert.empty(self.reference_models)
        else:
            Assert.eq(self.reference_models.keys(), {name})
        if self.model.base_model.use_absolute_position_embeddings:
            Assert.geq(self.model.base_model.num_absolute_position_embeddings, self.batch.sequence_length)
        # if self.model.base_model.distillation_model is not None:
        #     # TODO: Support loss masking for distillation?
        #     assert not self.batch.use_loss_masking_spans
        for reference_model in self.reference_models.values():
            Assert.none(reference_model.model.base_model.distillation_model)
            # TODO: Support more LM head features.
            Assert.none(reference_model.model.base_model.cross_entropy_splits)
            Assert.eq(reference_model.model.base_model.parallel_embeddings, self.model.base_model.parallel_embeddings)
            Assert.geq(reference_model.model.base_model.prediction_heads, self.model.base_model.prediction_heads)

    @classmethod
    def get_inference_runner_class(cls) -> type["GPTInferenceRunner"]:
        from fast_llm.models.gpt.model import GPTInferenceRunner

        # TODO: we dont have inference runner for SSM/Hybrid yet, should return None?
        logger.warning(
            "No inference runner for SSM/Hybrid yet, using GPTInferenceRunner for now, which does not support SSM/Hybrid"
        )

        return GPTInferenceRunner
