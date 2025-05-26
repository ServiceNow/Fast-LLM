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
from fast_llm.layers.ssm.config import SSMBlockType, SSMConfig, SSMDimNames
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.layers.transformer.transformer import BaseBlock, TransformerLayer
from fast_llm.models.gpt.config import GPTBatchConfig, PretrainedGPTModelConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.models.gpt.model import GPTInferenceRunner
    from fast_llm.models.hybrid.huggingface import HuggingfaceHybridSSMModelForCausalLM
    from fast_llm.models.hybrid.model import HybridSSMModel
    from fast_llm.models.hybrid.trainer import SSMTrainer

logger = logging.getLogger(__name__)


@config_class(registry=True)
class BlockConfig(Config):
    _abstract = True
    block_class: typing.ClassVar[type[BaseBlock]]
    # config: TransformerConfig | SSMConfig

    lr_scale: list[float] | None = Field(
        default=None,
        desc="Custom learning rate scale for each layer.",
        doc="May be used to freeze some layers by setting their scale to zero.",
        hint=FieldHint.feature,
    )

    def setup_tensor_space(self, tensor_space: "TensorSpace", block_name: str) -> None:
        raise NotImplementedError()

    @property
    def hidden_size(self) -> int:
        raise NotImplementedError()


@config_class(dynamic_type={BlockConfig: "transformer"})
class TransformerBlockConfig(BlockConfig, TransformerConfig):
    _abstract = False
    block_class: typing.ClassVar[type[BaseBlock]] = TransformerLayer

    def setup_tensor_space(self, tensor_space: "TensorSpace", block_name: str) -> None:
        TransformerConfig.setup_tensor_space(self, tensor_space, block_name)


@config_class(dynamic_type={BlockConfig: "discrete_mamba2"})
class DiscreteMamba2BlockConfig(BlockConfig, SSMConfig):
    _abstract = False
    block_class: typing.ClassVar[type[BaseBlock]] = LlambaBlock

    hidden_size: int = Field(
        default=1024,
        desc="Hidden size of the block.",
        hint=FieldHint.architecture,
    )

    # def _validate(self):
    #     self.config.validate()

    def setup_tensor_space(self, tensor_space: TensorSpace, block_name: str) -> None:

        d_inner = int(self.expansion_factor * self.hidden_size) if self.d_inner is None else self.d_inner
        # Hidden dimension
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


@config_class(dynamic_type={BlockConfig: "mamba"})
class MambaBlockConfig(BlockConfig, SSMConfig):
    _abstract = False
    block_class: typing.ClassVar[type[BaseBlock]] = LlambaOneBlock

    hidden_size: int = Field(
        default=1024,
        desc="Hidden size of the block.",
        hint=FieldHint.architecture,
    )

    def setup_tensor_space(self, tensor_space: TensorSpace, name: str) -> None:

        if self.dt_rank is None:
            mamba_dt_rank = math.ceil(self.hidden_size / 16)
        else:
            mamba_dt_rank = self.dt_rank

        d_inner = int(self.expansion_factor * self.hidden_size) if self.d_inner is None else self.d_inner
        # Hidden dimension
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.model_dim}_{name}", self.hidden_size))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.inner_dim}_{name}", d_inner))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.state_dim}_{name}", self.state_size))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.dt_rank}_{name}", mamba_dt_rank))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.x_proj_dim}_{name}", mamba_dt_rank + self.state_size * 2))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.conv_kernel_size}_{name}", self.conv_kernel_dimension))
        tensor_space.add_tensor_dim(TensorDim(f"{SSMDimNames.inner_proj_mamba}_{name}", d_inner * 2))


@config_class()
class HybridSSMBaseModelConfig(LanguageModelBaseConfig):
    _abstract = False
    ############################################################################################
    # Note, transformer and ssm are here for legacy reasons, we should migrate to blocks field
    transformer: TransformerConfig = Field(
        desc="Configuration for the transformer architecture. Note, having transformer and ssm fields in HybridSSMBaseModelConfig is depricated.",
        hint=FieldHint.architecture,
    )

    ssm: SSMConfig = Field(
        desc="Configuration for the SSM architecture. Note, having transformer and ssm fields in HybridSSMBaseModelConfig is depricated.",
        hint=FieldHint.architecture,
    )
    ############################################################################################
    blocks: dict[str, BlockConfig] = Field(
        default=None,
        desc="Named block configurations that can be referenced in block_pattern.",
        hint=FieldHint.architecture,
    )

    hybrid_block_layout: list[str] | None = Field(
        default=None,
        desc=f"Pattern of blocks to use in the model (still supports the previous depricated format with  {SSMBlockType.__members__.values()})",
        hint=FieldHint.core,
    )

    default_mtp_type: str | None = Field(
        default=None,
        desc="Multi-token prediction mixer to use in the model. 't' for Transformer, 'm' for Mamba1, 'm2' for discrete Mamba2. If None, will use the last block type in `hybrid_block_layout`.",
        hint=FieldHint.optional,
    )
    # TODO: ideally these things should be move to LanguageModelBaseConfig?
    # TODO: currently num_layers is defined in TransformerConfig, but ideally this should be migrated to LanguageModelBaseConfig in the future.
    # Hence, for now: the num_layers should be set in the first transformer block, if no transformer blocks used we will fallback to num_layers from here.
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
        if self.blocks is not None and self.hybrid_block_layout is not None:
            # Validate that all pattern entries refer to valid blocks
            for block_name in self.hybrid_block_layout:
                if block_name not in self.blocks:
                    raise ValueError(f"Block name '{block_name}' in block_pattern not found in blocks dictionary")

            first_transformer_block_config: TransformerBlockConfig | None = None

            for block_name, block_config in self.blocks.items():
                if isinstance(block_config, TransformerBlockConfig):
                    if first_transformer_block_config is None:
                        first_transformer_block_config = block_config
                    else:
                        logger.warning(
                            f"Found multiple transformer blocks with different number of layers, using num_layers from the first transformer block for all"
                        )
                block_config._validate()

            if first_transformer_block_config is not None:
                num_layers = first_transformer_block_config.config.num_layers
                logger.warning(
                    f"TransformerBlockConfig overwrites BaseModelConfig num_layers, setting num_layers = {num_layers}"
                )
                self.num_layers = num_layers
        else:
            logger.warning(
                f"No transformer blocks found in blocks dictionary, using num_layers from BaseModelConfig: {self.num_layers} and falling back to old behavior with hybrid_block_layout containing any of {SSMBlockType.__members__.values()}"
            )
            if self.hybrid_block_layout is None:
                with self._set_implicit_default():
                    self.hybrid_block_layout = [SSMBlockType.mamba2_discrete.value]

            if len(self.hybrid_block_layout) != self.transformer.num_layers:
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
            Assert.custom(
                lambda _: all(
                    block_type in SSMBlockType.__members__.values() for block_type in self.hybrid_block_layout
                ),
                f"Invalid block type: {self.hybrid_block_layout}. Must be one of {SSMBlockType.__members__.values()}",
            )
            Assert.custom(
                lambda _: self.default_mtp_type in SSMBlockType.__members__.values() or self.default_mtp_type is None,
                f"Invalid MTP type: {self.default_mtp_type}. Must be one of {SSMBlockType.__members__.values()} or None",
            )
            # TODO: prepare hybrid_block_layout here

        with self._set_implicit_default():
            if self.init_method_std_embed is None:
                self.init_method_std_embed = (
                    first_transformer_block_config.config.init_method_std
                    if first_transformer_block_config is not None
                    else 0.02
                )
            if self.init_method_max_embed is None:
                self.init_method_max_embed = (
                    first_transformer_block_config.config.init_method_max
                    if first_transformer_block_config is not None
                    else 0.02
                )
            if self.init_method_min_embed is None:
                self.init_method_min_embed = (
                    first_transformer_block_config.config.init_method_min
                    if first_transformer_block_config is not None
                    else 0.02
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
class HybridSSMModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "hybrid_ssm"
    base_model: HybridSSMBaseModelConfig = FieldUpdate()
    checkpoint_formats = FastLLMModelConfig.checkpoint_formats + (
        LLambaHuggingfaceCheckpointFormat,
        AprielSSMHuggingfaceCheckpointFormat,
        AprielSSMHHybridHuggingfaceCheckpointFormat,
    )

    @classmethod
    def get_model_class(cls) -> type["HybridSSMModel"]:
        from fast_llm.models.hybrid.model import HybridSSMModel

        return HybridSSMModel

    @classmethod
    def get_huggingface_model_class(cls) -> type["HuggingfaceHybridSSMModelForCausalLM"]:
        from fast_llm.models.hybrid.huggingface import HuggingfaceHybridSSMModelForCausalLM

        return HuggingfaceHybridSSMModelForCausalLM

    def _validate(self):
        logger.warning(
            "HybridSSMModelConfig is being instantiated. This model is experimental and may not work as expected."
        )
        super()._validate()


@config_class()
class PretrainedHybridSSMModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: HybridSSMModelConfig = FieldUpdate()


@config_class()
class HybridTrainerConfig(PretrainedHybridSSMModelConfig, TrainerConfig):
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
