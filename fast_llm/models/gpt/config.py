import functools
import logging
import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointHandler
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.schedule.config import BatchConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.models.gpt.megatron import set_megatron_distributed_seeds
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM
    from fast_llm.models.gpt.model import GPTInferenceRunner, GPTModel
    from fast_llm.models.gpt.trainer import GPTTrainer

logger = logging.getLogger(__name__)


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


class DiffusionDreamGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "dream"
    trust_remote_code: typing.ClassVar[bool] = True


class DiffusionLlamaGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "diffusion_llama"
    trust_remote_code: typing.ClassVar[bool] = True


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
    truncate_documents: bool | None = Field(
        default=True,
        desc=(
            "If enabled, documents may be truncated while being packed to fit the sequence length."
            "Otherwise, sequences will be padded such that every document lies entirely within a sample"
            " (and documents exceeding the sequence length will be skipped altogether)."
        ),
        hint=FieldHint.feature,
    )

    def _validate(self) -> None:
        if self.micro_sequence_length is None:
            with self._set_implicit_default():
                self.micro_sequence_length = self.sequence_length
        super()._validate()

    @functools.cached_property
    def micro_batch_splits(self) -> int:
        assert self._validated
        return div(self.sequence_length, self.micro_sequence_length)


@config_class()
class GPTBaseModelConfig(LanguageModelBaseConfig):
    _abstract = False

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
        if "transposed_mlp_weight" in default:
            assert default.pop("transposed_mlp_weight")
        if "match_megatron" in default:
            assert "use_megatron_initialization" not in default
            default["use_megatron_initialization"] = default.pop("match_megatron")
        if "layer_norm_impl" in default:
            assert "normalization_implementation" not in default
            default["normalization_implementation"] = default.pop("layer_norm_impl")
        if "fused_mlp" in default:
            del default["fused_mlp"]
        return super()._from_dict(default, strict, flat)


@config_class(dynamic_type={FastLLMModelConfig: "gpt"})
class GPTModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "gpt"
    base_model: GPTBaseModelConfig = FieldUpdate()
    checkpoint_formats: typing.ClassVar[tuple[type[CheckpointFormat], ...]] = FastLLMModelConfig.checkpoint_formats + (
        AutoGPTHuggingfaceCheckpointFormat,
        Starcoder2GPTHuggingfaceCheckpointFormat,
        LlamaGPTHuggingfaceCheckpointFormat,
        Qwen2GPTHuggingfaceCheckpointFormat,
        MistralGPTHuggingfaceCheckpointFormat,
        MixtralGPTHuggingfaceCheckpointFormat,
        MTPLlamaGPTHuggingfaceCheckpointFormat,
        DiffusionDreamGPTHuggingfaceCheckpointFormat,
        DiffusionLlamaGPTHuggingfaceCheckpointFormat,
    )

    @classmethod
    def get_model_class(cls) -> type["GPTModel"]:
        from fast_llm.models.gpt.model import GPTModel

        return GPTModel

    @classmethod
    def get_huggingface_model_for_causal_lm_class(cls) -> type["HuggingfaceGPTModelForCausalLM"]:
        from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

        return HuggingfaceGPTModelForCausalLM


@config_class()
class PretrainedGPTModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: GPTModelConfig = FieldUpdate()


@config_class(dynamic_type={RunnableConfig: "train_gpt", TrainerConfig: "gpt"})
class GPTTrainerConfig(PretrainedGPTModelConfig, TrainerConfig):
    data: GPTDataConfig = FieldUpdate()
    batch: GPTBatchConfig = FieldUpdate()
    # TODO: Use dynamic model type?
    reference_models: dict[str, PretrainedGPTModelConfig] = FieldUpdate()

    def _validate(self) -> None:
        if self.batch.sequence_length is None:
            # TODO: Drop this.
            self.batch.sequence_length = self.model.base_model.max_position_embeddings
        if self.model.base_model.use_megatron_initialization:
            set_megatron_distributed_seeds(self.model.distributed)
        super()._validate()

        if self.model.base_model.use_absolute_position_embeddings:
            Assert.geq(self.model.base_model.num_absolute_position_embeddings, self.batch.sequence_length)

        distillation_model = self.model.base_model.distillation_model
        dpo_reference_model = self.model.base_model.dpo_reference_model

        if self.model.base_model.enable_dpo:
            assert dpo_reference_model is not None
            Assert.none(distillation_model)
        else:
            Assert.none(dpo_reference_model)

        if distillation_model is None and dpo_reference_model is None:
            Assert.empty(self.reference_models)
        else:
            assert distillation_model is None or dpo_reference_model is None  # currently don't support both
            expected_names = {name for name in (distillation_model, dpo_reference_model) if name is not None}
            Assert.eq(self.reference_models.keys(), expected_names)

        for reference_model in self.reference_models.values():
            Assert.none(reference_model.model.base_model.distillation_model)
            Assert.none(reference_model.model.base_model.dpo_reference_model)
            # TODO: Support more LM head features.
            Assert.none(reference_model.model.base_model.cross_entropy_splits)
            Assert.eq(reference_model.model.base_model.parallel_embeddings, self.model.base_model.parallel_embeddings)
            Assert.geq(reference_model.model.base_model.prediction_heads, self.model.base_model.prediction_heads)

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
        if "truncate_documents" in default.get("data", {}):
            # Backward compatibility for the legacy truncate_documents field.
            # TODO v0.x: Remove backward compatibility.
            logger.warning(
                "`data.truncate_documents` field is deprecated. " "Please use `batch.truncate_documents` instead."
            )
            assert "truncate_documents" not in default.get("batch", {})
            if "batch" not in default:
                default["batch"] = {}
            default["batch"]["truncate_documents"] = default["data"].pop("truncate_documents")
        return super()._from_dict(default, strict, flat)

    @classmethod
    def get_trainer_class(cls) -> type["GPTTrainer"]:
        from fast_llm.models.gpt.trainer import GPTTrainer

        return GPTTrainer

    @classmethod
    def get_inference_runner_class(cls) -> type["GPTInferenceRunner"]:
        from fast_llm.models.gpt.model import GPTInferenceRunner

        return GPTInferenceRunner
