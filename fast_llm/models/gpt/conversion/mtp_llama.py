import functools
import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    NestedWeightConverter,
    OutputProjectionWeightConverter,
    RenameConfigConverter,
    WeightConverter,
)
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.config import MTPLlamaCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    LlamaBaseModelConverter,
    LlamaHeadConverter,
    LlamaHuggingfaceCheckpointHandler,
)
from fast_llm.utils import safe_merge_dicts


class MTPLlamaHeadConverter(LlamaHeadConverter):
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
        # MTP-Llama places the first prediction head's final norm under ``model.mtp_norms.0`` instead
        # of the standard ``model.norm``; the additional MTP blocks/norms come from the imperative
        # ``get_converters`` override below since their count depends on ``head.prediction_heads``.
        return {
            "final_norm": NestedWeightConverter(
                "final_norm", "model.mtp_norms.0", cls.normalization_converter_class, config_attr="normalization"
            ),
            "output_weights": OutputProjectionWeightConverter("output_weights", "lm_head.weight"),
        }

    @classmethod
    def get_converters(
        cls,
        config: GPTBaseModelConfig,
        exported_config: dict,
    ) -> list[WeightConverter]:
        converters = list(cls.emit_weight_converters(config.head, "head", "", root_config=config))
        # Append the MTP fan-out: one block + one norm per extra prediction head. ``block_converter_class``
        # comes from the parent ``LlamaHeadConverter`` ClassVar — the MTP block shape matches the main
        # decoder block.
        for prediction_distance in range(2, config.head.prediction_heads + 1):
            converters += cls.block_converter_class.emit_weight_converters(
                config.decoder.last_block_config,
                f"multi_token_prediction.blocks.{prediction_distance - 2}",
                f"model.mtp_heads.{prediction_distance - 2}",
                root_config=config,
            )
            converters += cls.normalization_converter_class.emit_weight_converters(
                config.head.normalization,
                f"multi_token_prediction.heads.{prediction_distance - 2}.final_norm",
                f"model.mtp_norms.{prediction_distance - 1}",
                root_config=config,
            )
        return converters


class MTPLlamaBaseModelConverter(LlamaBaseModelConverter):
    head_converter_class: typing.ClassVar[type[MTPLlamaHeadConverter]] = MTPLlamaHeadConverter


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
