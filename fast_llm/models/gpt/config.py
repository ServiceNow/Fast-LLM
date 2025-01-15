import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.language_model.config import LanguageModelArchitectureConfig, LanguageModelBaseConfig
from fast_llm.models.gpt.megatron import set_megatron_distributed_seeds

if typing.TYPE_CHECKING:
    from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
    from fast_llm.models.gpt.model import GPTModel
    from fast_llm.models.gpt.trainer import GPTTrainer


class GPTHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.gpt.conversion import AutoGPTHuggingfaceCheckpointHandler

        return AutoGPTHuggingfaceCheckpointHandler.get_handler_class(cls.name)


class AutoGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "auto"


class Starcoder2GPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "starcoder2"


class LlamaGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "llama"


class MistralGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "mistral"


class MixtralGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "mixtral"


@config_class()
class GPTArchitectureConfig(LanguageModelArchitectureConfig):
    _abstract = False

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        # TODO v0.3: Remove backward compatibility fix
        if "transposed_mlp_weight" in default:
            assert default.pop("transposed_mlp_weight")
        return super()._from_dict(default, strict, flat)


@config_class()
class GPTBaseModelConfig(LanguageModelBaseConfig, GPTArchitectureConfig):
    architecture_class = GPTArchitectureConfig

    # Debug, to get an exact match with megatron init.
    use_megatron_initialization: bool = Field(
        default=False, desc="Exactly match the initialization of a Megatron model.", hint=FieldHint.testing
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


@config_class()
class GPTModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "gpt"
    base_model: GPTBaseModelConfig = FieldUpdate(default_factory=GPTBaseModelConfig)
    checkpoint_formats: typing.ClassVar[tuple[type[CheckpointFormat], ...]] = FastLLMModelConfig.checkpoint_formats + (
        AutoGPTHuggingfaceCheckpointFormat,
        Starcoder2GPTHuggingfaceCheckpointFormat,
        LlamaGPTHuggingfaceCheckpointFormat,
        MistralGPTHuggingfaceCheckpointFormat,
        MixtralGPTHuggingfaceCheckpointFormat,
    )

    @classmethod
    def get_model_class(cls) -> type["GPTModel"]:
        from fast_llm.models.gpt.model import GPTModel

        return GPTModel

    @classmethod
    def get_huggingface_model_class(cls) -> type["HuggingfaceGPTModelForCausalLM"]:
        from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

        return HuggingfaceGPTModelForCausalLM


@config_class()
class PretrainedGPTModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: GPTModelConfig = FieldUpdate(default_factory=GPTModelConfig)


@config_class()
class GPTTrainerConfig(PretrainedGPTModelConfig, TrainerConfig):
    data: GPTDataConfig = FieldUpdate(default_factory=GPTDataConfig)

    def _validate(self) -> None:
        if self.batch.sequence_length is None:
            # TODO: Drop this.
            self.batch.sequence_length = self.model.base_model.max_position_embeddings
        if self.model.base_model.use_megatron_initialization:
            set_megatron_distributed_seeds(self.model.distributed)
        super()._validate()

    @classmethod
    def get_trainer_class(cls) -> type["GPTTrainer"]:
        from fast_llm.models.gpt.trainer import GPTTrainer

        return GPTTrainer
