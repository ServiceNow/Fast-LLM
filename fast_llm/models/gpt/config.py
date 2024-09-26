import typing

from fast_llm.config import ConfigDictFormat, Field, FieldHint, config_class
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.language_model.config import LanguageModelArchitectureConfig, LanguageModelBaseConfig
from fast_llm.models.gpt.megatron import set_megatron_distributed_seeds

if typing.TYPE_CHECKING:
    from fast_llm.engine.multi_stage.conversion import ModelConverter


@config_class()
class GPTArchitectureConfig(LanguageModelArchitectureConfig):
    _abstract = False

    @classmethod
    def from_dict(
        cls,
        arg_dict: dict,
        format_: ConfigDictFormat = ConfigDictFormat.flat,
        strict: bool = True,
        strict_cls: bool = False,
    ):
        # Backward compatibility
        if "transposed_mlp_weight" in arg_dict:
            assert arg_dict.pop("transposed_mlp_weight")
        return super().from_dict(arg_dict, format_=format_, strict=strict, strict_cls=strict_cls)

    @classmethod
    def get_converter_class(cls, model_type: str | None = None) -> type["ModelConverter"]:
        from fast_llm.models.gpt.conversion import AutoGPTConverter

        return AutoGPTConverter if model_type is None else AutoGPTConverter.converter_map[model_type]


@config_class()
class GPTBaseModelConfig(LanguageModelBaseConfig, GPTArchitectureConfig):
    architecture_cls = GPTArchitectureConfig

    # Debug, to get an exact match with megatron init.
    use_megatron_initialization: bool = Field(
        default=False, desc="Exactly match the initialization of a Megatron model.", hint=FieldHint.testing
    )

    @classmethod
    def from_dict(
        cls,
        arg_dict: dict,
        format_: ConfigDictFormat = ConfigDictFormat.flat,
        strict: bool = True,
        strict_cls: bool = False,
    ):
        # Backward compatibility
        if "match_megatron" in arg_dict:
            assert "use_megatron_initialization" not in arg_dict
            arg_dict["use_megatron_initialization"] = arg_dict.pop("match_megatron")
        if "layer_norm_impl" in arg_dict:
            assert "normalization_implementation" not in arg_dict
            arg_dict["normalization_implementation"] = arg_dict.pop("layer_norm_impl")
        if "fused_mlp" in arg_dict:
            del arg_dict["fused_mlp"]
        return super().from_dict(arg_dict, format_=format_, strict=strict, strict_cls=strict)


@config_class()
class GPTModelConfig(FastLLMModelConfig):
    _abstract = False
    base_model: GPTBaseModelConfig = Field(default_factory=GPTBaseModelConfig)

    @classmethod
    def get_model_class(cls):
        from fast_llm.models.gpt.model import GPTModel

        return GPTModel

    @classmethod
    def get_huggingface_model_class(cls):
        from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

        return HuggingfaceGPTModelForCausalLM


@config_class()
class PretrainedGPTModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: GPTModelConfig = Field(default_factory=GPTModelConfig)


@config_class()
class GPTTrainerConfig(PretrainedGPTModelConfig, TrainerConfig):
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


class HuggingfaceModelType:
    """
    An enum for the huggingface models with conversion support.
    """

    starcoder2 = "starcoder2"
    llama = "llama"
    mistral = "mistral"
    mixtral = "mixtral"
