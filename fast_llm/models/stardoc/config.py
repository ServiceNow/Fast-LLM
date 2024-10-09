import typing

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.language_model.config import LanguageModelArchitectureConfig, LanguageModelBaseConfig

if typing.TYPE_CHECKING:
    from fast_llm.engine.multi_stage.conversion import ModelConverter


@config_class()
class StarDocArchitectureConfig(LanguageModelArchitectureConfig):
    _abstract = False

    @classmethod
    def _from_dict(
        cls,
        default: dict,
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove backward compatibility fix
        if "transposed_mlp_weight" in default:
            assert default.pop("transposed_mlp_weight")
        return super()._from_dict(default, strict, flat)

    @classmethod
    def get_converter_class(cls, model_type: str | None = None) -> type["ModelConverter"]:
        from fast_llm.models.stardoc.conversion import AutoStarDocConverter

        return AutoStarDocConverter if model_type is None else AutoStarDocConverter.converter_map[model_type]


@config_class()
class StarDocBaseModelConfig(LanguageModelBaseConfig, StarDocArchitectureConfig):
    architecture_cls = StarDocArchitectureConfig

    @classmethod
    def _from_dict(
        cls,
        default: dict,
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove backward compatibility fix
        if "layer_norm_impl" in default:
            assert "normalization_implementation" not in default
            default["normalization_implementation"] = default.pop("layer_norm_impl")
        if "fused_mlp" in default:
            del default["fused_mlp"]
        return super()._from_dict(default, strict, flat)


@config_class()
class StarDocModelConfig(FastLLMModelConfig):
    _abstract = False
    base_model: StarDocBaseModelConfig = Field(default_factory=StarDocBaseModelConfig)

    @classmethod
    def get_model_class(cls):
        from fast_llm.models.stardoc.model import StarDocModel

        return StarDocModel

    @classmethod
    def get_huggingface_model_class(cls):
        from fast_llm.models.stardoc.huggingface import HuggingfaceStarDocModelForCausalLM

        return HuggingfaceStarDocModelForCausalLM


@config_class()
class PretrainedStarDocModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: StarDocModelConfig = Field(default_factory=StarDocModelConfig)


@config_class()
class StarDocTrainerConfig(PretrainedStarDocModelConfig, TrainerConfig):
    def _setup(self):
        super()._setup()
        if self.batch.sequence_length is None:
            # TODO: Drop this.
            self.batch.sequence_length = self.base_model.max_position_embeddings

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