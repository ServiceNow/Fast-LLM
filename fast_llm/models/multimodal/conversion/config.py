import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointHandler


class MultimodalHuggingfaceCheckpointFormat(CheckpointFormat):
    support_optimizer: typing.ClassVar[bool] = False

    @classmethod
    def get_handler_class(cls) -> type[CheckpointHandler]:
        from fast_llm.models.multimodal.conversion.auto import AutoMultimodalHuggingfaceCheckpointHandler

        return AutoMultimodalHuggingfaceCheckpointHandler.get_handler_class(cls.name)


class AutoMultimodalHuggingfaceCheckpointFormat(MultimodalHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "auto"


class LlavaCheckpointFormat(MultimodalHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "llava"


class LlavaHybridSSMCheckpointFormat(MultimodalHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "llava_hybrid_ssm"


class Apriel2CheckpointFormat(MultimodalHuggingfaceCheckpointFormat):
    name: typing.ClassVar[str] = "apriel2"
