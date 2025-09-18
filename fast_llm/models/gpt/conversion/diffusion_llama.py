import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import DiffusionLlamaCheckpointFormat
from fast_llm.models.gpt.conversion.llama import LlamaHuggingfaceCheckpointHandler
from fast_llm.utils import safe_merge_dicts


class DiffusionLlamaHuggingfaceCheckpointHandler(LlamaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = DiffusionLlamaCheckpointFormat
    architecture: typing.ClassVar[str] = "DiffusionLlamaModel"

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.diffusion_llama.configuration_diffusion_llama import DiffusionLlamaConfig

        return DiffusionLlamaConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.diffusion_llama import (
            configuration_diffusion_llama,
            generation_utils,
            modeling_diffusion_llama,
        )

        return configuration_diffusion_llama.__file__, modeling_diffusion_llama.__file__, generation_utils.__file__

    @classmethod
    def _export_config(cls, config: GPTModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_diffusion_llama.DiffusionLlamaConfig",
                    "AutoModel": "modeling_diffusion_llama.DiffusionLlamaModel",
                },
            },
        )
