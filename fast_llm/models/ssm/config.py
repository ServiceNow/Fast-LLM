import logging
import math
import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointHandler
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.ssm.config import SSMBlockType, SSMConfig, SSMDimNames
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTBatchConfig, PretrainedGPTModelConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.models.ssm.huggingface import HuggingfaceHybridSSMModelForCausalLM
    from fast_llm.models.ssm.model import HybridSSMInferenceRunner, HybridSSMModel
    from fast_llm.models.ssm.trainer import HybridSSMTrainer

logger = logging.getLogger(__name__)


@config_class()
class HybridSSMBaseModelConfig(GPTBaseModelConfig):
    _abstract = False

    ssm: SSMConfig = Field(
        desc="Configuration for the transformer architecture.",
        hint=FieldHint.architecture,
    )
    hybrid_block_layout: list[SSMBlockType] | None = Field(
        default=None,
        desc=f"Pattern of blocks to use in the model. Available types: {SSMBlockType.__members__.values()}",
        hint=FieldHint.core,
    )
    default_mtp_type: SSMBlockType | None = Field(
        default=None,
        desc="Multi-token prediction mixer to use in the model. If None, will use the last block type in `hybrid_block_layout`.",
        hint=FieldHint.optional,
    )
    # TODO: Support combination of different SSM block types.
    ssm_block_type: SSMBlockType | None = Field(init=False)

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        """
        Setup the tensor space for the model.
        Some of these can be setup directly in the layer config, but keeping them here for clarity.
        """
        super().setup_tensor_space(tensor_space)
        d_inner: int = self.ssm.d_inner

        # Hidden dimension
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.model_dim, self.transformer.hidden_size))
        # Mamba-specific dimensions
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.inner_dim, d_inner))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.state_dim, self.ssm.state_size))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.dt_rank, self.ssm.dt_rank))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.x_proj_dim, self.ssm.dt_rank + self.ssm.state_size * 2))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.conv_kernel_size, self.ssm.conv_kernel_dimension))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.inner_proj_mamba, d_inner * 2))

        if SSMBlockType.mamba2_discrete.value in self.hybrid_block_layout:
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
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.inner_proj_discrete_mamba2, inner_proj_dim))
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.conv_dim, conv_dim))
        elif SSMBlockType.mamba2.value in self.hybrid_block_layout:
            inner_proj_dim: int = 2 * self.ssm.d_xb + 2 * d_inner  # + self.ssm.dt_rank
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.inner_proj_mamba2, inner_proj_dim))
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.x_proj_dim_2, self.ssm.d_xb))
            tensor_space.add_tensor_dim(TensorDim(SSMDimNames.c_heads, d_inner // self.ssm.state_size))

    def _validate(self):
        with self._set_implicit_default(None):
            if self.ssm.dt_rank == "auto" or self.ssm.dt_rank is None:
                self.ssm.dt_rank = math.ceil(self.transformer.hidden_size / 16)
        with self._set_implicit_default():
            if self.ssm.d_xb is None:
                self.ssm.d_xb = self.transformer.hidden_size
            if self.ssm.d_inner is None:
                self.ssm.d_inner = int(self.ssm.expansion_factor * self.transformer.hidden_size)

        if self.hybrid_block_layout is None:
            with self._set_implicit_default():
                self.hybrid_block_layout = [SSMBlockType.mamba2_discrete] * self.transformer.num_layers

        if len(self.hybrid_block_layout) != self.transformer.num_layers:
            message = f"hybrid_block_layout length {len(self.hybrid_block_layout)} does not match num_layers {self.transformer.num_layers}"
            if self.transformer.num_layers % len(self.hybrid_block_layout) != 0:
                raise ValueError(message)
            num_repeats = self.transformer.num_layers // len(self.hybrid_block_layout)
            logger.warning(f"{message}, will repeat {self.hybrid_block_layout} {num_repeats} times.")
            self.hybrid_block_layout = self.hybrid_block_layout * num_repeats

        super()._validate()
        ssm_block_types = set(self.hybrid_block_layout) - {SSMBlockType.transformer}
        # TODO: Support combination of different SSM block types.
        Assert.leq(len(ssm_block_types), 1)
        self.ssm_block_type = ssm_block_types.pop() if ssm_block_types else None


class LLambaHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    name: typing.ClassVar[str] = "llamba"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import LLambaHuggingfaceCheckpointHandler

        return LLambaHuggingfaceCheckpointHandler


class AprielSSMHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    name: typing.ClassVar[str] = "apriel_ssm"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import AprielSSMHuggingfaceCheckpointHandler

        return AprielSSMHuggingfaceCheckpointHandler


class AprielSSMHHybridHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    name: typing.ClassVar[str] = "apriel_ssm_hybrid"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import AprielSSMHHybridHuggingfaceCheckpointHandler

        return AprielSSMHHybridHuggingfaceCheckpointHandler


class AprielThinkerSSMHHybridHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    name: typing.ClassVar[str] = "apriel_ssm_thinker_hybrid"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import AprielThinkerSSMHHybridHuggingfaceCheckpointHandler

        return AprielThinkerSSMHHybridHuggingfaceCheckpointHandler


@config_class(dynamic_type={FastLLMModelConfig: "hybrid_ssm"})
class HybridSSMModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "hybrid_ssm"
    base_model: HybridSSMBaseModelConfig = FieldUpdate()
    checkpoint_formats = FastLLMModelConfig.checkpoint_formats + (
        LLambaHuggingfaceCheckpointFormat,
        AprielSSMHuggingfaceCheckpointFormat,
        AprielSSMHHybridHuggingfaceCheckpointFormat,
        AprielThinkerSSMHHybridHuggingfaceCheckpointFormat,
    )

    @classmethod
    def get_model_class(cls) -> type["HybridSSMModel"]:
        from fast_llm.models.ssm.model import HybridSSMModel

        return HybridSSMModel

    @classmethod
    def get_inference_runner_class(cls) -> type["HybridSSMInferenceRunner"]:
        from fast_llm.models.ssm.model import HybridSSMInferenceRunner

        logger.warning(
            "HybridSSMInferenceRunner only supports training-style forward pass. Use generate with cache disabled."
        )

        return HybridSSMInferenceRunner

    @classmethod
    def get_huggingface_model_for_causal_lm_class(cls) -> type["HuggingfaceHybridSSMModelForCausalLM"]:
        from fast_llm.models.ssm.huggingface import HuggingfaceHybridSSMModelForCausalLM

        return HuggingfaceHybridSSMModelForCausalLM

    def _validate(self):
        logger.warning(
            "HybridSSMModelConfig is being instantiated. This model is experimental and may not work as expected."
        )
        if (
            self.base_model.sequence_first
            or self.distributed.sequence_data_parallel > 1
            or self.distributed.sequence_tensor_parallel
        ):
            raise NotImplementedError(f"Sequence-first not supported for SSMs.")
        super()._validate()


@config_class()
class PretrainedHybridSSMModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: HybridSSMModelConfig = FieldUpdate()


@config_class(dynamic_type={RunnableConfig: "train_hybrid_ssm", TrainerConfig: "hybrid_ssm"})
class HybridSSMTrainerConfig(PretrainedHybridSSMModelConfig, TrainerConfig):
    data: GPTDataConfig = FieldUpdate()
    batch: GPTBatchConfig = FieldUpdate()
    reference_models: dict[str, PretrainedGPTModelConfig] = FieldUpdate()

    @classmethod
    def get_trainer_class(cls) -> type["HybridSSMTrainer"]:
        from fast_llm.models.ssm.trainer import HybridSSMTrainer

        return HybridSSMTrainer

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
