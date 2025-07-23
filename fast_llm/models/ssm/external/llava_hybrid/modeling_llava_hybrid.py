from torch import nn
from transformers import LlavaForConditionalGeneration, LlavaModel

from .configuration_llava_hybrid import LlavaHybridConfig


class LlavaHybridModel(LlavaModel):
    """
    Llava SSM-Hybrid-decoder model.
    """

    def __init__(self, config: LlavaHybridConfig):
        super().__init__(config)


class LlavaHybridForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: LlavaHybridConfig):
        super(LlavaForConditionalGeneration, self).__init__(config)
        self.model = LlavaHybridModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
