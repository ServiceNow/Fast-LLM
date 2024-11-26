import typing

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.data.config import DataConfig
from fast_llm.engine.training.config import TrainerConfig

from fast_llm.models.gpt.config import (
    GPTArchitectureConfig,
    GPTBaseModelConfig,
    GPTTrainerConfig,
)

from fast_llm.layers.multimodal_model.config import MultimodalModelArchitectureConfig, MultimodalModelBaseConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace

if typing.TYPE_CHECKING:
    from fast_llm.engine.multi_stage.conversion import ModelConverter


@config_class()
class StarDocDataConfig(DataConfig):
    # TODO: If needed, inherit from AbstractDataConfig instead and re-implement everything.
    pass


@config_class()
class StarDocArchitectureConfig(GPTArchitectureConfig):
    multimodal_model: MultimodalModelArchitectureConfig = Field(
        default_factory=MultimodalModelArchitectureConfig,
        desc="Configuration for the multimodal components (image encoder and adapter).",
        hint=FieldHint.core,
    )

    def setup_tensor_space(self, tensor_space: TensorSpace):
        super().setup_tensor_space(tensor_space)
        self.multimodal_model.setup_tensor_space(tensor_space)
    
    @classmethod
    def get_converter_class(cls, model_type: str | None = None) -> type["ModelConverter"]:
        from fast_llm.models.stardoc.conversion import AutoStarDocConverter

        return AutoStarDocConverter if model_type is None else AutoStarDocConverter.converter_map[model_type]

@config_class()
class StarDocBaseModelConfig(GPTBaseModelConfig, StarDocArchitectureConfig):
    architecture_cls = StarDocArchitectureConfig

    multimodal_model: MultimodalModelBaseConfig = Field(
        default_factory=MultimodalModelBaseConfig,
        desc="Configuration for the multimodal components (image encoder and adapter).",
        hint=FieldHint.core,
    )



@config_class()
class StarDocModelConfig(FastLLMModelConfig):
    _abstract = False
    base_model: StarDocBaseModelConfig = Field(default_factory=StarDocBaseModelConfig)

    @classmethod
    def get_model_class(cls):
        from fast_llm.models.stardoc.model import StarDocModel

        return StarDocModel


@config_class()
class PretrainedStarDocModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: StarDocModelConfig = Field(default_factory=StarDocModelConfig)


@config_class()
class StarDocTrainerConfig(PretrainedStarDocModelConfig, GPTTrainerConfig):
    @classmethod
    def get_trainer_class(cls):
        from fast_llm.models.stardoc.trainer import StarDocTrainer

        return StarDocTrainer


class HuggingfaceModelType:
    """
    An enum for the huggingface models with conversion support.
    """

    starcoder2 = "starcoder2"
    llama = "llama"
    mistral = "mistral"
    mixtral = "mixtral"