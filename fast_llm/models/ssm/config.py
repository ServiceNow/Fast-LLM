import logging
import typing

from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.config import Field, FieldHint, FieldUpdate, config_class
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTArchitectureConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig


if typing.TYPE_CHECKING:
    from fast_llm.models.ssm.model import HybridModel
    from fast_llm.models.gpt.trainer import GPTTrainer

logger = logging.getLogger(__name__)

class HybridArchitectureConfig(GPTArchitectureConfig):
    pass


@config_class()
class HybridBaseModelConfig(LanguageModelBaseConfig, HybridArchitectureConfig):
    architecture_class = HybridArchitectureConfig

    # Debug, to get an exact match with megatron init.
    use_megatron_initialization: bool = Field(
        default=False, desc="Exactly match the initialization of a Megatron model.", hint=FieldHint.testing
    )

    block_pattern: list[str] = Field(
        default_factory=list,
        desc="Pattern of blocks to use in the model. 't' for Transformer, 'm' for Mamba.",
        hint=FieldHint.core,
    )
    
    # Mamba configuration parameters
    mamba_expansion_factor: int =  Field(
        default=2,
        desc="Expansion factor for Mamba blocks.",
        hint=FieldHint.core,
    )
    mamba_state_size: int = Field(
        default=16,
        desc="State size for Mamba blocks.",
        hint=FieldHint.core,
    )
    mamba_conv_dimension: int = Field(
        default=4,
        desc="Conv dimension for Mamba blocks.",
        hint=FieldHint.core,
    )
    mamba_rms_norm: bool = Field(
        default=True,
        desc="Use RMS normalization for Mamba blocks.",
        hint=FieldHint.core,
    )

    mamba_residual_in_fp32: bool = Field(
        default=True,
        desc="Use residual in fp32 for Mamba blocks.",
        hint=FieldHint.core,
    )
    mamba_fused_add_norm: bool = Field(
        default=False,
        desc="Use fused add norm for Mamba blocks.",
        hint=FieldHint.core,
    )
    mamba_layernorm_epsilon: float = Field(
        default=1e-5,
        desc="Epsilon for layer normalization for Mamba blocks.",
        hint=FieldHint.core,
    )

    use_fast_path: bool = Field(
        default=False,
        desc="Use fast path for Mamba blocks.",
        hint=FieldHint.core,
    )

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

