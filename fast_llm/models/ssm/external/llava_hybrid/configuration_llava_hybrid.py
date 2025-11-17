from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING
from transformers.utils import logging

try:
    # In the fast-llm repo, import from the SSM modeling file
    from fast_llm.models.ssm.external.apriel_15b_hybrid.configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig
except ImportError:
    # In the exported checkpoint, import from local file
    from .configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig

logger = logging.get_logger(__name__)

# Copied from configuration_ssm_hybrid_apriel15b.py
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


class LlavaHybridConfig(PretrainedConfig):
    """
    Configuration class for Llava SSM-Hybrid-decoder model.
    """

    model_type = "llava_hybrid"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=32000,
        projector_hidden_act="gelu",
        projector_intermediate_size=4096,
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_seq_length=576,
        multimodal_projector_bias=True,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        # projector_intermediate_size is an addition to the original Llava config
        self.projector_intermediate_size = projector_intermediate_size
        self.image_seq_length = image_seq_length

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            # Load the custom SSM hybrid config if specified
            if text_config.get("model_type") == "apriel_ssm_thinker_hybrid":
                text_config = AprielSSMHybridConfig(**text_config)
            else:
                text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
                text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias

        super().__init__(**kwargs)


__all__ = ["LlavaHybridConfig"]
