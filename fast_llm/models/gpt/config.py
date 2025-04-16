import typing
from functools import cached_property

from fast_llm.config import Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.language_model.config import LanguageModelArchitectureConfig, LanguageModelBaseConfig
from fast_llm.models.gpt.megatron import set_megatron_distributed_seeds
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
    from fast_llm.models.gpt.model import GPTInferenceRunner, GPTModel
    from fast_llm.models.gpt.trainer import GPTTrainer


class GPTHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False
    trust_remote_code: typing.ClassVar[bool] = False

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


class Qwen2GPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "qwen2"


class MistralGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "mistral"


class MixtralGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "mixtral"


class MTPLlamaGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "mtp_llama"
    trust_remote_code: typing.ClassVar[bool] = True


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
class GPTBatchConfig(BatchConfig):
    sequence_length: int = Field(
        default=2048,
        desc="Number of tokens in a sample.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    micro_sequence_length: int = Field(
        default=None,
        desc="Number of tokens in a micro-sequence (must divide the sequence length).",
        hint=FieldHint.performance,
        valid=check_field(Assert.gt, 0),
    )
    # TODO: Find a better place for these?
    cross_document_attention: bool = Field(
        default=True,
        desc="Applies attention to tokens from other documents in the packed sequence. Set to False for masking attention to other documents.",
        hint=FieldHint.feature,
    )
    use_loss_masking_spans: bool = Field(
        default=False,
        desc="Read loss masking spans from the dataset.",
        hint=FieldHint.feature,
    )

    def _validate(self) -> None:
        if self.micro_sequence_length is None:
            with self._set_implicit_default():
                self.micro_sequence_length = self.sequence_length
        super()._validate()

    @cached_property
    def micro_batch_splits(self) -> int:
        return div(self.sequence_length, self.micro_sequence_length)


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
        Qwen2GPTHuggingfaceCheckpointFormat,
        MistralGPTHuggingfaceCheckpointFormat,
        MixtralGPTHuggingfaceCheckpointFormat,
        MTPLlamaGPTHuggingfaceCheckpointFormat,
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
    batch: GPTBatchConfig = FieldUpdate(default_factory=GPTBatchConfig)
    # TODO: Use dynamic model type?
    reference_models: dict[str, PretrainedGPTModelConfig] = FieldUpdate()

    def _validate(self) -> None:
        if self.batch.sequence_length is None:
            # TODO: Drop this.
            self.batch.sequence_length = self.model.base_model.max_position_embeddings
        if self.model.base_model.use_megatron_initialization:
            set_megatron_distributed_seeds(self.model.distributed)
        for reference_model in self.reference_models.values():
            Assert.none(reference_model.model.base_model.cross_entropy_splits)
        super()._validate()

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ) -> typing.Self:
        # TODO v0.x: Remove backward compatibility.
        cls._handle_renamed_field(
            default, ("data", "sampling", "use_loss_masking_spans"), ("batch", "use_loss_masking_spans")
        )
        return super()._from_dict(default, strict, flat)

    @classmethod
    def get_trainer_class(cls) -> type["GPTTrainer"]:
        from fast_llm.models.gpt.trainer import GPTTrainer

        return GPTTrainer

    @classmethod
    def get_inference_runner_class(cls) -> type["GPTInferenceRunner"]:
        from fast_llm.models.gpt.model import GPTInferenceRunner

        return GPTInferenceRunner
