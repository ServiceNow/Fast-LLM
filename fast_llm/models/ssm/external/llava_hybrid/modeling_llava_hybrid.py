from transformers import LlavaModel

from .configuration_llava_hybrid import LlavaHybridConfig


class LlavaHybridModel(LlavaModel):
    """
    Llava SSM-Hybrid-decoder model.
    """

    def __init__(self, config: LlavaHybridConfig):
        super().__init__(config)
