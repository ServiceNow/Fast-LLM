import logging
import math
import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointHandler
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.layers.ssm.config import SSMConfig, SSMDimNames
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.models.ssm.huggingface import HuggingfaceHybridSSMModelForCausalLM
    from fast_llm.models.ssm.model import HybridSSMModel
    from fast_llm.models.ssm.trainer import SSMTrainer

logger = logging.getLogger(__name__)


@config_class()
class HybridSSMBaseModelConfig(LanguageModelBaseConfig):
    _abstract = False

    ssm: SSMConfig = Field(
        desc="Configuration for the transformer architecture.",
        hint=FieldHint.architecture,
    )
    hybrid_block_layout: list[str] = Field(
        default_factory=lambda: ["m2"],
        desc="Pattern of blocks to use in the model. 't' for Transformer, 'm' for Mamba1, 'm2' for Discrete Mamba2",
        hint=FieldHint.architecture,
    )
    default_mtp_type: str | None = Field(
        default=None,
        desc="Multi-token prediction mixer to use in the model. 't' for Transformer, 'm' for Mamba1, 'm2' for discrete Mamba2. If None, will use the last block type in `hybrid_block_layout`.",
        hint=FieldHint.optional,
    )
    use_megatron_initialization: bool = Field(
        default=False, desc="Exactly match the initialization of a Megatron model.", hint=FieldHint.testing
    )  # TODO: is this needed?

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        """
        Setup the tensor space for the model.
        Some of these can be setup directly in the layer config, but keeping them here for clarity.
        """
        super().setup_tensor_space(tensor_space)
        if not "m2" in self.hybrid_block_layout and not "m" in self.hybrid_block_layout:
            raise ValueError(
                "Block pattern must contain at least one 'm' or 'm2', use gpt model for transformer only architectures"
            )

        if self.ssm.dt_rank < 0:
            mamba_dt_rank = math.ceil(self.transformer.hidden_size / 16)
        else:
            mamba_dt_rank = self.ssm.dt_rank

        d_inner = int(self.ssm.expansion_factor * self.transformer.hidden_size)
        # Hidden dimension
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.model_dim, self.transformer.hidden_size))
        # Mamba-specific dimensions
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.inner_dim, d_inner))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.state_dim, self.ssm.state_size))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.dt_rank, mamba_dt_rank))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.x_proj_dim, mamba_dt_rank + self.ssm.state_size * 2))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.conv_kernel_size, self.ssm.conv_kernel_dimension))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.inner_proj_mamba, d_inner * 2))

        if "m2" in self.hybrid_block_layout or self.default_mtp_type == "m2":
            # Mamba2 specific dimensions
            # as per https://github.com/cartesia-ai/edge/blob/a0e121ebed3d2324c6d762b0e211a08d62583681/cartesia-pytorch/cartesia_pytorch/Llamba/mixers/discrete_mamba2.py#L66C3-L66C4
            headdim = d_inner // self.ssm.n_v_heads
            Assert.eq(self.ssm.n_v_heads, d_inner // headdim)
            Assert.eq(d_inner % headdim, 0)
            Assert.eq(self.ssm.n_v_heads % self.ssm.n_qk_heads, 0)

            conv_dim = d_inner + 2 * self.ssm.n_qk_heads * self.ssm.state_size
            inner_proj_dim = 2 * d_inner + 2 * self.ssm.n_qk_heads * self.ssm.state_size + self.ssm.n_v_heads

            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.head_dim, headdim))
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.qk_heads, self.ssm.n_qk_heads))
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.v_heads, self.ssm.n_v_heads))
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.inner_proj_mamba2, inner_proj_dim))
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.conv_dim, conv_dim))

    def _validate(self):
        len_block_layout = len(self.hybrid_block_layout)
        if len_block_layout != self.transformer.num_layers:
            if self.transformer.num_layers % len_block_layout != 0:
                raise ValueError(
                    f"hybrid_block_layout length {len_block_layout} does not match num_layers {self.transformer.num_layers}"
                )
            num_repeats = int(self.transformer.num_layers // len_block_layout)
            logger.warning(
                f"hybrid_block_layout length {len_block_layout} does not match num_layers {self.transformer.num_layers}, will repeat {self.hybrid_block_layout} {num_repeats} times"
            )
            self.hybrid_block_layout = self.hybrid_block_layout * num_repeats

        Assert.eq(len_block_layout, self.transformer.num_layers)
        Assert.custom(
            lambda _: all(block_type in ["t", "m", "m2"] for block_type in self.hybrid_block_layout),
            f"Invalid block type: {self.hybrid_block_layout}. Must be 't' or 'm' or 'm2'",
        )
        Assert.custom(
            lambda _: self.default_mtp_type in ["t", "m", "m2", None],
            f"Invalid MTP type: {self.default_mtp_type}. Must be 't' or 'm' or 'm2' or None",
        )

        super()._validate()


class LLambaHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    name: typing.ClassVar[str] = "llamba"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import LLambaHuggingfaceCheckpointHandler

        return LLambaHuggingfaceCheckpointHandler


@config_class()
class HybridSSMModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "hybrid_ssm"
    base_model: HybridSSMBaseModelConfig = FieldUpdate()
    checkpoint_formats = FastLLMModelConfig.checkpoint_formats + (LLambaHuggingfaceCheckpointFormat,)

    @classmethod
    def get_model_class(cls) -> type["HybridSSMModel"]:
        from fast_llm.models.ssm.model import HybridSSMModel

        return HybridSSMModel

    @classmethod
    def get_huggingface_model_class(cls) -> type["HuggingfaceHybridSSMModelForCausalLM"]:
        from fast_llm.models.ssm.huggingface import HuggingfaceHybridSSMModelForCausalLM

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

    @classmethod
    def get_trainer_class(cls) -> type["SSMTrainer"]:
        from fast_llm.models.ssm.trainer import SSMTrainer

        return SSMTrainer
