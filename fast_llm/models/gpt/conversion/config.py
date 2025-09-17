import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointHandler


class GPTHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.gpt.conversion.auto import AutoGPTHuggingfaceCheckpointHandler

        return AutoGPTHuggingfaceCheckpointHandler.get_handler_class(cls.name)


class AutoGPTHuggingfaceCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "auto"


class LlamaCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "llama"


class Qwen2CheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "qwen2"


class MistralCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "mistral"


class MixtralCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "mixtral"


class MTPLlamaCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "mtp_llama"


class DiffusionDreamCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "dream"


class DiffusionLlamaCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "diffusion_llama"


class AprielHybridSSMCheckpointFormat(GPTHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "apriel_hybrid_ssm"
