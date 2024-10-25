import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, config_class
from fast_llm.data.config import DataConfig
from fast_llm.engine.checkpoint.config import Converter
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.language_model.config import LanguageModelArchitectureConfig, LanguageModelBaseConfig
from fast_llm.models.gpt.megatron import set_megatron_distributed_seeds

if typing.TYPE_CHECKING:
    pass


class HuggingfaceModelType:
    """
    An enum for the huggingface models with conversion support.
    """

    auto = "auto"
    starcoder2 = "starcoder2"
    llama = "llama"
    mistral = "mistral"
    mixtral = "mixtral"


@config_class()
class GPTArchitectureConfig(LanguageModelArchitectureConfig):
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


@config_class()
class GPTBaseModelConfig(LanguageModelBaseConfig, GPTArchitectureConfig):
    architecture_cls = GPTArchitectureConfig

    # Debug, to get an exact match with megatron init.
    use_megatron_initialization: bool = Field(
        default=False, desc="Exactly match the initialization of a Megatron model.", hint=FieldHint.testing
    )

    @classmethod
    def _from_dict(
        cls,
        default: dict,
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove backward compatibility fix
        if "match_megatron" in default:
            assert "use_megatron_initialization" not in default
            default["use_megatron_initialization"] = default.pop("match_megatron")
        if "layer_norm_impl" in default:
            assert "normalization_implementation" not in default
            default["normalization_implementation"] = default.pop("layer_norm_impl")
        if "fused_mlp" in default:
            del default["fused_mlp"]
        return super()._from_dict(default, strict, flat)


@config_class()
class GPTModelConfig(FastLLMModelConfig):
    _abstract = False
    base_model: GPTBaseModelConfig = FieldUpdate(default_factory=GPTBaseModelConfig)

    @classmethod
    def get_model_class(cls):
        from fast_llm.models.gpt.model import GPTModel

        return GPTModel

    @classmethod
    def get_huggingface_model_class(cls):
        from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

        return HuggingfaceGPTModelForCausalLM

    @classmethod
    def get_supported_checkpoint_formats(cls):
        return super().get_supported_checkpoint_formats() + tuple(
            name for name in HuggingfaceModelType.__dict__ if not name.startswith("_")
        )

    @classmethod
    def get_converter_class(cls, format: str) -> type["Converter"]:
        try:
            return super().get_converter_class(format)
        except NotImplementedError:
            from fast_llm.models.gpt.conversion import AutoGPTConverter

            return AutoGPTConverter.get_converter_class(format)


@config_class()
class PretrainedGPTModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: GPTModelConfig = FieldUpdate(default_factory=GPTModelConfig)


@config_class()
class GPTTrainerConfig(PretrainedGPTModelConfig, TrainerConfig):

    data: DataConfig = FieldUpdate(default_factory=DataConfig)

    def _setup(self):
        super()._setup()
        if self.batch.sequence_length is None:
            # TODO: Drop this.
            self.batch.sequence_length = self.base_model.max_position_embeddings
        if self.base_model.use_megatron_initialization:
            set_megatron_distributed_seeds(self.distributed)

    @classmethod
    def get_trainer_class(cls):
        from fast_llm.models.gpt.trainer import GPTTrainer

        return GPTTrainer
