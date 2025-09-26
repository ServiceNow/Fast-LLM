import abc

from fast_llm.engine.checkpoint.external import AutoStateDictCheckpointHandler
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.models.gpt.conversion.apriel import AprielHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.config import (
    AprielHybridSSMCheckpointFormat,
    DiffusionDreamCheckpointFormat,
    DiffusionLlamaCheckpointFormat,
    LlamaCheckpointFormat,
    MistralCheckpointFormat,
    MixtralCheckpointFormat,
    MTPLlamaCheckpointFormat,
    Qwen2CheckpointFormat,
)
from fast_llm.models.gpt.conversion.diffusion_dream import DiffusionDreamHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.diffusion_llama import DiffusionLlamaHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.llama import LlamaHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.mistral import MistralHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.mixtral import MixtralHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.mtp_llama import MTPLlamaHuggingfaceCheckpointHandler
from fast_llm.models.gpt.conversion.qwen2 import Qwen2HuggingfaceCheckpointHandler


class AutoGPTHuggingfaceCheckpointHandler(
    AutoStateDictCheckpointHandler, HuggingfaceStateDictCheckpointHandler, abc.ABC
):

    handler_map = {
        LlamaCheckpointFormat.name: LlamaHuggingfaceCheckpointHandler,
        Qwen2CheckpointFormat.name: Qwen2HuggingfaceCheckpointHandler,
        MistralCheckpointFormat.name: MistralHuggingfaceCheckpointHandler,
        MixtralCheckpointFormat.name: MixtralHuggingfaceCheckpointHandler,
        MTPLlamaCheckpointFormat.name: MTPLlamaHuggingfaceCheckpointHandler,
        DiffusionDreamCheckpointFormat.name: DiffusionDreamHuggingfaceCheckpointHandler,
        DiffusionLlamaCheckpointFormat.name: DiffusionLlamaHuggingfaceCheckpointHandler,
        AprielHybridSSMCheckpointFormat.name: AprielHuggingfaceCheckpointHandler,
    }
