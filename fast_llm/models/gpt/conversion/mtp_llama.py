import functools
import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    NestedWeightConverter,
    OutputProjectionWeightConverter,
    RenameConfigConverter,
    RepeatWeightConverter,
    WeightConverter,
)
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import MTPLlamaCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaBaseModelConverter,
    LlamaBlockConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
)
from fast_llm.utils import safe_merge_dicts


class MTPLlamaHeadConverter(LlamaHeadConverter):
    # The MTP block shape matches the main decoder block, so we plug ``LlamaBlockConverter`` in directly.
    block_converter_class: typing.ClassVar[type[LlamaBlockConverter]] = LlamaBlockConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # MTP-Llama exposes the prediction-heads count via the HF config; Llama itself blanket-ignores it.
            "prediction_heads": RenameConfigConverter(("prediction_heads",), ("prediction_heads",)),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # MTP-Llama places the first prediction head's final norm under ``model.mtp_norms.0`` instead of
        # the standard ``model.norm``. The additional per-prediction-distance blocks and norms are
        # declared on the base-model converter (they live at the model root, not under ``head``).
        return {
            "final_norm": NestedWeightConverter(
                "final_norm", "model.mtp_norms.0", cls.normalization_converter_class, config_attr="normalization"
            ),
            "output_weights": OutputProjectionWeightConverter("output_weights", "lm_head.weight"),
        }


class MTPLlamaBaseModelConverter(LlamaBaseModelConverter):
    head_converter_class: typing.ClassVar[type[MTPLlamaHeadConverter]] = MTPLlamaHeadConverter

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # The extra prediction-distance heads (distances 2..prediction_heads) repeat the main decoder's
        # last block and the head normalization. They sit at the model root as ``multi_token_prediction.*``,
        # interleaved on the HF side with the base head's ``model.mtp_norms.0`` (declared on the head).
        return {
            **super()._create_weight_converters(),
            "multi_token_prediction_blocks": RepeatWeightConverter(
                cls.block_converter_class,
                count=lambda config: config.head.prediction_heads - 1,
                sub_config=lambda config: config.decoder.last_block_config,
                fast_llm_prefix=lambda index: f"multi_token_prediction.blocks.{index}",
                hf_prefix=lambda index: f"model.mtp_heads.{index}",
            ),
            "multi_token_prediction_norms": RepeatWeightConverter(
                cls.head_converter_class.normalization_converter_class,
                count=lambda config: config.head.prediction_heads - 1,
                sub_config=lambda config: config.head.normalization,
                fast_llm_prefix=lambda index: f"multi_token_prediction.heads.{index}.final_norm",
                hf_prefix=lambda index: f"model.mtp_norms.{index + 1}",
            ),
        }


class MTPLlamaHuggingfaceCheckpointHandler(LlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MTPLlamaCheckpointFormat
    architecture: typing.ClassVar[str] = "MTPLlamaForCausalLM"
    base_model_converter_class: typing.ClassVar[type[MTPLlamaBaseModelConverter]] = MTPLlamaBaseModelConverter

    @classmethod
    def _export_config(cls, config: GPTModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_mtp_llama.MTPLlamaConfig",
                    "AutoModel": "modeling_mtp_llama.MTPLlamaModel",
                    "AutoModelForCausalLM": "modeling_mtp_llama.MTPLlamaForCausalLM",
                },
            },
        )

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.mtp_llama.configuration_mtp_llama import MTPLlamaConfig

        return MTPLlamaConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.mtp_llama import configuration_mtp_llama, modeling_mtp_llama

        return configuration_mtp_llama.__file__, modeling_mtp_llama.__file__, None
