import logging
import typing

from fast_llm.config import Field, FieldHint, FieldUpdate, config_class
from fast_llm.data.data.gpt.config import GPTDataConfig
from fast_llm.engine.checkpoint.config import CheckpointHandler
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, PretrainedFastLLMModelConfig
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.ssm.config import SSMBlockType, SSMConfig
from fast_llm.models.gpt.config import (
    GPTBaseModelConfig,
    GPTBatchConfig,
    GPTHuggingfaceCheckpointFormat,
    PretrainedGPTModelConfig,
)
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.models.ssm.huggingface import HuggingfaceHybridSSMModelForCausalLM
    from fast_llm.models.ssm.model import HybridSSMInferenceRunner, HybridSSMModel
    from fast_llm.models.ssm.trainer import HybridSSMTrainer

logger = logging.getLogger(__name__)


@config_class()
class HybridSSMBaseModelConfig(GPTBaseModelConfig):
    _abstract = False

    ssm: SSMConfig = Field(
        desc="Configuration for the transformer architecture.",
        hint=FieldHint.architecture,
    )
    hybrid_block_layout: list[SSMBlockType] | None = Field(
        default=None,
        desc=f"Pattern of blocks to use in the model. Available types: {SSMBlockType.__members__.values()}",
        hint=FieldHint.core,
    )
    default_mtp_type: SSMBlockType | None = Field(
        default=None,
        desc="Multi-token prediction mixer to use in the model. If None, will use the last block type in `hybrid_block_layout`.",
        hint=FieldHint.optional,
    )
    # TODO: Support combination of different SSM block types.
    ssm_block_type: SSMBlockType | None = Field(init=False)

    def _validate(self):
        self.ssm.set_defaults(self.transformer.hidden_size)

        if self.hybrid_block_layout is None:
            with self._set_implicit_default():
                self.hybrid_block_layout = [SSMBlockType.mamba2_discrete] * self.transformer.num_layers

        if len(self.hybrid_block_layout) != self.transformer.num_layers:
            message = f"hybrid_block_layout length {len(self.hybrid_block_layout)} does not match num_layers {self.transformer.num_layers}"
            if self.transformer.num_layers % len(self.hybrid_block_layout) != 0:
                raise ValueError(message)
            num_repeats = self.transformer.num_layers // len(self.hybrid_block_layout)
            logger.warning(f"{message}, will repeat {self.hybrid_block_layout} {num_repeats} times.")
            self.hybrid_block_layout = self.hybrid_block_layout * num_repeats

        super()._validate()
        ssm_block_types = set(self.hybrid_block_layout) - {SSMBlockType.transformer}
        # TODO: Support combination of different SSM block types.
        Assert.leq(len(ssm_block_types), 1)
        self.ssm_block_type = ssm_block_types.pop() if ssm_block_types else None


class LLambaHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "llamba"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import LLambaHuggingfaceCheckpointHandler

        return LLambaHuggingfaceCheckpointHandler


class AprielSSMHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "apriel_ssm"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import AprielSSMHuggingfaceCheckpointHandler

        return AprielSSMHuggingfaceCheckpointHandler


class AprielSSMHHybridHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "apriel_ssm_hybrid"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import AprielSSMHHybridHuggingfaceCheckpointHandler

        return AprielSSMHHybridHuggingfaceCheckpointHandler


class AprielThinkerSSMHHybridHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "apriel_ssm_thinker_hybrid"

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.ssm.conversion import AprielThinkerSSMHHybridHuggingfaceCheckpointHandler

        return AprielThinkerSSMHHybridHuggingfaceCheckpointHandler


@config_class(dynamic_type={FastLLMModelConfig: "hybrid_ssm"})
class HybridSSMModelConfig(FastLLMModelConfig):
    _abstract = False
    model_name: typing.ClassVar[str] = "hybrid_ssm"
    base_model: HybridSSMBaseModelConfig = FieldUpdate()
    checkpoint_formats = FastLLMModelConfig.checkpoint_formats + (
        LLambaHuggingfaceCheckpointFormat,
        AprielSSMHuggingfaceCheckpointFormat,
        AprielSSMHHybridHuggingfaceCheckpointFormat,
        AprielThinkerSSMHHybridHuggingfaceCheckpointFormat,
    )

    @classmethod
    def get_model_class(cls) -> type["HybridSSMModel"]:
        from fast_llm.models.ssm.model import HybridSSMModel

        return HybridSSMModel

    @classmethod
    def get_inference_runner_class(cls) -> type["HybridSSMInferenceRunner"]:
        from fast_llm.models.ssm.model import HybridSSMInferenceRunner

        logger.warning(
            "HybridSSMInferenceRunner only supports training-style forward pass. Use generate with cache disabled."
        )

        return HybridSSMInferenceRunner

    @classmethod
    def get_huggingface_model_for_causal_lm_class(cls) -> type["HuggingfaceHybridSSMModelForCausalLM"]:
        from fast_llm.models.ssm.huggingface import HuggingfaceHybridSSMModelForCausalLM

        return HuggingfaceHybridSSMModelForCausalLM

    def _validate(self):
        logger.warning(
            "HybridSSMModelConfig is being instantiated. This model is experimental and may not work as expected."
        )
        super()._validate()


@config_class()
class PretrainedHybridSSMModelConfig(PretrainedFastLLMModelConfig):
    _abstract = False
    model: HybridSSMModelConfig = FieldUpdate()


@config_class(dynamic_type={RunnableConfig: "train_hybrid_ssm", TrainerConfig: "hybrid_ssm"})
class HybridSSMTrainerConfig(PretrainedHybridSSMModelConfig, TrainerConfig):
    data: GPTDataConfig = FieldUpdate()
    batch: GPTBatchConfig = FieldUpdate()
    reference_models: dict[str, PretrainedGPTModelConfig] = FieldUpdate()

    @classmethod
    def get_trainer_class(cls) -> type["HybridSSMTrainer"]:
        from fast_llm.models.ssm.trainer import HybridSSMTrainer

        return HybridSSMTrainer

    def _validate(self) -> None:
        super()._validate()
        if (name := self.model.base_model.distillation_model) is None:
            Assert.empty(self.reference_models)
        else:
            Assert.eq(self.reference_models.keys(), {name})
        if self.model.base_model.use_absolute_position_embeddings:
            Assert.geq(self.model.base_model.num_absolute_position_embeddings, self.batch.sequence_length)
        # if self.model.base_model.distillation_model is not None:
        #     # TODO: Support loss masking for distillation?
        #     assert not self.batch.use_loss_masking_spans
        for reference_model in self.reference_models.values():
            Assert.none(reference_model.model.base_model.distillation_model)
            # TODO: Support more LM head features.
            Assert.none(reference_model.model.base_model.cross_entropy_splits)
            Assert.eq(reference_model.model.base_model.parallel_embeddings, self.model.base_model.parallel_embeddings)
            Assert.geq(reference_model.model.base_model.prediction_heads, self.model.base_model.prediction_heads)
