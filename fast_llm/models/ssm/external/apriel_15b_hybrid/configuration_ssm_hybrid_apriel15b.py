from transformers import MistralConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)
# TODO: split into mamba 2 and discrete mamba 2 configs with a base dict
ssm_config_default = {
    # discrete mamba2
    "d_state": 64,
    "n_v_heads": 32,
    "n_qk_heads": 32,
    "expand": 1,
    "chunk_size": 128,
    "activation": "identity",
    "bias": False,
    "d_conv": 4,
    "d_inner": 32 * 128,
    # mamba2
    "d_xb": None,  # will be set to model dim
    "dt_rank": "auto",
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init": "random",
    "dt_scale": 1.0,
    "dt_init_floor": 1e-4,
    "conv_bias": True,
}


class AprielGDNConfig:
    def __init__(
        self,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        kl_short_conv_kernel_size=4,
        kl_num_heads=32,
        kl_head_dim=128,
    ):
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        # Kimi LInear
        self.short_conv_kernel_size = kl_short_conv_kernel_size
        self.head_dim = kl_head_dim
        self.num_heads = kl_num_heads


LAYER_TYPES = {"t": "full_attention", "swa": "sliding_attention", "gdn": "gated_delta_net", "kl": "kimi_linear"}


class AprielSSMHybridConfig(MistralConfig):
    model_type = "apriel_ssm_thinker_hybrid"

    def __init__(self, hybrid_block_layout=["t"], ssm_cfg=None, gdn_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_block_layout = hybrid_block_layout
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads  # as in transformers 4.51.3
        self.ssm_cfg = ssm_cfg or ssm_config_default

        gdn_config: AprielGDNConfig = (
            AprielGDNConfig(**gdn_cfg) if isinstance(gdn_cfg, dict) else gdn_cfg or AprielGDNConfig()
        )

        # make elements of gdn_config accessible as attributes of self to pass self directly to Qwen3NextGatedDeltaNet
        for k, v in vars(gdn_config).items():
            setattr(self, k, v)

        for k, v in ssm_config_default.items():
            if k not in self.ssm_cfg:
                self.ssm_cfg[k] = v  # to make sure all elements are present in the config
        self.layer_types = [LAYER_TYPES[lt] for lt in hybrid_block_layout]  # this is for vllm compatibility
        self.linear_attn_config = {
            "short_conv_kernel_size": gdn_config.short_conv_kernel_size,
            "head_dim": gdn_config.head_dim,
            "num_heads": gdn_config.num_heads,
        }
