from transformers import MistralConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

ssm_config_default = {
    "d_state": 64,
    "n_v_heads": 32,
    "n_qk_heads": 32,
    "expand": 1,
    "chunk_size": 128,
    "activation": "identity",
    "bias": False,
    "d_conv": 4,
    "d_inner": 32 * 128,
}


class AprielSSMHybridConfig(MistralConfig):
    model_type = "apriel_ssm_thinker_hybrid"

    def __init__(self, hybrid_block_layout=["m2d"], ssm_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_block_layout = hybrid_block_layout
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads  # as in transformers 4.51.3
        self.ssm_cfg = ssm_cfg or ssm_config_default

        for k, v in ssm_config_default.items():
            if k not in self.ssm_cfg:
                self.ssm_cfg[k] = v  # to make sure all elements are present in the config
