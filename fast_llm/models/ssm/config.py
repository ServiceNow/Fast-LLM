import logging
import typing

from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.config import Field, FieldHint, FieldUpdate, config_class
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.models.gpt.config import GPTArchitectureConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.tensor import TensorSpace, TensorDim
from fast_llm.layers.ssm.config import SSMDimNames, SSMArchitectureConfig
import math

if typing.TYPE_CHECKING:
    from fast_llm.models.ssm.model import HybridModel

logger = logging.getLogger(__name__)

class HybridArchitectureConfig(GPTArchitectureConfig):
    pass


@config_class()
class HybridBaseModelConfig(LanguageModelBaseConfig, HybridArchitectureConfig):
    architecture_class = HybridArchitectureConfig

    ssm: SSMArchitectureConfig = Field(
        default_factory=SSMArchitectureConfig,
        desc="Configuration for the transformer architecture.",
        hint=FieldHint.core,
    )

    # Debug, to get an exact match with megatron init.
    use_megatron_initialization: bool = Field(
        default=False, desc="Exactly match the initialization of a Megatron model.", hint=FieldHint.testing
    )

    block_pattern: list[str] = Field(
        default_factory=list,
        desc="Pattern of blocks to use in the model. 't' for Transformer, 'm' for Mamba.",
        hint=FieldHint.core,
    )

    use_fast_path: bool = Field(
        default=True,
        desc="Use fast path for Mamba blocks.",
        hint=FieldHint.core,
    )

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        super().setup_tensor_space(tensor_space)
        # tensor = tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        if self.ssm.dt_rank == "auto":
            mamba_dt_rank = math.ceil(self.transformer.hidden_size / 16)
        else:
            mamba_dt_rank = self.ssm.dt_rank
        
        # Hidden dimension
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.d_model, self.transformer.hidden_size))        
        # Mamba-specific dimensions
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.d_inner, int(self.ssm.expansion_factor * self.transformer.hidden_size)))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.d_state, self.ssm.state_size))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.d_conv, self.ssm.conv_dimension))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.dt_rank,  mamba_dt_rank))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.d_inner_2, self.ssm.expansion_factor * self.transformer.hidden_size * 2))
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.d_x_proj, mamba_dt_rank + self.ssm.state_size * 2))

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        # TODO v0.3: Remove backward compatibility fix
        if "match_megatron" in default:
            assert "use_megatron_initialization" not in default
            default["use_megatron_initialization"] = default.pop("match_megatron")
        if "layer_norm_impl" in default:
            assert "normalization_implementation" not in default
            default["normalization_implementation"] = default.pop("layer_norm_impl")
        if "fused_mlp" in default:
            del default["fused_mlp"]
        return super()._from_dict(default, strict, flat)
    
    def __post_init__(self):
        super().__post_init__()
        if len(self.block_pattern) == 0:
            logger.warning("No block pattern provided, using default pattern of Transformer blocks.")
            self.block_pattern = ['t'] * self.transformer.num_layers
    
    def _validate(self):
        super()._validate()


@config_class()
class HybridModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "hybrid_ssm"
    base_model: HybridBaseModelConfig = FieldUpdate(default_factory=HybridBaseModelConfig)

    @classmethod
    def get_model_class(cls) -> type["HybridModel"]:
        from fast_llm.models.ssm.model import HybridModel

        return HybridModel


@config_class()
class PretrainedHybridModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: HybridModelConfig = FieldUpdate(default_factory=HybridModelConfig)


@config_class()
class HybridTrainerConfig(PretrainedHybridModelConfig, TrainerConfig):
    data: GPTDataConfig = FieldUpdate(default_factory=GPTDataConfig)

    @classmethod
    def get_trainer_class(cls) -> type["SSMTrainer"]:
        from fast_llm.models.ssm.trainer import SSMTrainer

        return SSMTrainer

