from fast_llm.config import Field, FieldUpdate, config_class
from fast_llm.data.config import DataConfig
from fast_llm.models.gpt.config import (
    GPTArchitectureConfig,
    GPTBaseModelConfig,
    GPTModelConfig,
    GPTTrainerConfig,
    PretrainedGPTModelConfig,
)


@config_class()
class GRPOConfig:
    epsilon: float = Field(default=0.2, desc="PPO clipping parameter")
    kl_coef: float = Field(default=0.1, desc="KL divergence coefficient")
    ratio_threshold: float = Field(default=1.5, desc="Early stopping ratio threshold")
    use_advantages: bool = Field(default=True, desc="Use advantages instead of raw rewards")


@config_class()
class GRPODataConfig(DataConfig):
    # TODO: If needed, inherit from AbstractDataConfig instead and re-implement everything.
    pass


@config_class()
class GRPOArchitectureConfig(GPTArchitectureConfig):
    # TODO: Add custom base model architecture config parameters, if any.
    pass


@config_class()
class GRPOBaseModelConfig(GPTBaseModelConfig, GRPOArchitectureConfig):
    # TODO: Add custom other base model config parameters, if any.
    architecture_cls = GRPOArchitectureConfig
    grpo: GRPOConfig = Field(default_factory=GRPOConfig, desc="GRPO specific configuration")


@config_class()
class GRPOModelConfig(GPTModelConfig):
    # TODO: Add custom model config parameters, if any (typically none).
    base_model: GRPOBaseModelConfig = FieldUpdate(default_factory=GRPOBaseModelConfig)

    @classmethod
    def get_model_class(cls):
        from fast_llm.models.grpo.model import GRPOModel

        return GRPOModel

    @classmethod
    def get_huggingface_model_class(cls):
        from fast_llm.models.grpo.huggingface import HuggingfaceGRPOModelForCausalLM

        return HuggingfaceGRPOModelForCausalLM


@config_class()
class PretrainedGRPOModelConfig(PretrainedGPTModelConfig):
    model: GRPOModelConfig = FieldUpdate(default_factory=GRPOModelConfig)


@config_class()
class GRPOTrainerConfig(PretrainedGRPOModelConfig, GPTTrainerConfig):
    # TODO: Add custom trainer config parameters, if any (typically none).

    data: GRPODataConfig = FieldUpdate(default_factory=GRPODataConfig)

    @classmethod
    def get_trainer_class(cls):
        from fast_llm.models.grpo.trainer import GRPOTrainer

        return GRPOTrainer
