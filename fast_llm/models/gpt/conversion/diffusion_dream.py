import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import DiffusionDreamCheckpointFormat
from fast_llm.models.gpt.conversion.qwen2 import Qwen2HuggingfaceCheckpointHandler
from fast_llm.utils import safe_merge_dicts


class DiffusionDreamHuggingfaceCheckpointHandler(Qwen2HuggingfaceCheckpointHandler):
    """
    Handler for DiffusionDream Huggingface checkpoints.
    Inherits from Qwen2HuggingfaceCheckpointHandler (and CustomModelingExportMixin),
    but overrides _create_config_converters to update architectures and auto_map.
    """

    format: typing.ClassVar[type[CheckpointFormat]] = DiffusionDreamCheckpointFormat
    architecture: typing.ClassVar[str] = "DreamModel"

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.diffusion_dream.configuration_dream import DreamConfig

        return DreamConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.diffusion_dream import configuration_dream, generation_utils, modeling_dream

        return configuration_dream.__file__, modeling_dream.__file__, generation_utils.__file__

    @classmethod
    def _export_config(cls, config: GPTModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_dream.DreamConfig",
                    "AutoModel": "modeling_dream.DreamModel",
                },
            },
        )
