import abc

from fast_llm.engine.checkpoint.external import AutoStateDictCheckpointHandler
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.models.multimodal.conversion.apriel2 import Apriel2HuggingfaceCheckpointHandler
from fast_llm.models.multimodal.conversion.config import (
    Apriel2CheckpointFormat,
    LlavaCheckpointFormat,
    LlavaHybridSSMCheckpointFormat,
)
from fast_llm.models.multimodal.conversion.llava import LlavaHuggingfaceCheckpointHandler
from fast_llm.models.multimodal.conversion.llava_hybrid import LlavaHybridSSMHuggingfaceCheckpointHandler


class AutoMultimodalHuggingfaceCheckpointHandler(
    AutoStateDictCheckpointHandler, HuggingfaceStateDictCheckpointHandler, abc.ABC
):

    handler_map = {
        LlavaCheckpointFormat.name: LlavaHuggingfaceCheckpointHandler,
        LlavaHybridSSMCheckpointFormat.name: LlavaHybridSSMHuggingfaceCheckpointHandler,
        Apriel2CheckpointFormat.name: Apriel2HuggingfaceCheckpointHandler,
    }
