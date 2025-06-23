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

    def __init__(self, hybrid_block_layout=["m2d"], ssm_cfg=None, prediction_heads=1, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_block_layout = hybrid_block_layout
        self.set_if_mtp_model_type(prediction_heads)
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        self.prediction_heads = prediction_heads
        self.ssm_cfg = ssm_cfg or ssm_config_default

        for key, value in ssm_config_default.items():
            if key not in self.ssm_cfg:
                logger.warning(f"SSM config key '{key}' not found in provided ssm_cfg. Using default value: {value}")
                self.ssm_cfg[key] = value

    @classmethod
    def set_if_mtp_model_type(cls, prediction_heads):
        """
        Set the model type for the configuration.
        """
        if prediction_heads > 1:
            cls.model_type = "mtp_apriel_ssm_thinker_hybrid"
