from transformers import MistralConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AprielSSMHybridConfig(MistralConfig):
    model_type = "apriel_ssm_thinker_hybrid"

    def __init__(self, hybrid_block_layout=["m2d"], ssm_cfg=None, prediction_heads=1, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_block_layout = hybrid_block_layout
        self.prediction_heads = prediction_heads
        self.ssm_cfg = ssm_cfg or {
            "d_state": 64,
            "n_v_heads": 24,
            "n_qk_heads": 24,
            "expand": 1,
            "chunk_size": 128,
            "activation": "identity",
            "bias": False,
            "d_conv": 4,
            "d_inner": 24 * self.head_dim,  # num_heads * head_dim
        }
