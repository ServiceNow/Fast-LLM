import torch
import transformers

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.attention.rotary.config import Rotary2DConfig
from fast_llm.layers.vision.config import VisionKwargs
from fast_llm.utils import Assert
from tests.utils.utils import requires_cuda


@requires_cuda
def test_rotary_2d():
    """
    Compare Fast-LLM's implementation of 2d rotary embeddings with Pixtral.
    """
    head_dim = 16
    num_heads = 8

    patch_positions = torch.tensor(
        [[h, w] for h in range(4) for w in range(4)],
        dtype=torch.int64,
        device="cuda",
    )
    query = torch.empty(2, len(patch_positions), num_heads, head_dim, dtype=torch.float32, device="cuda").normal_()
    key = torch.empty_like(query).normal_()

    pixtral_config = transformers.PixtralVisionConfig(hidden_size=head_dim * num_heads, num_attention_heads=num_heads)
    pixtral_rotary = transformers.models.pixtral.modeling_pixtral.PixtralRotaryEmbedding(pixtral_config).to("cuda")
    # Convert patch positions (h, w) to Pixtral's linear position IDs
    # Pixtral expects: position_id = h * max_patches_per_side + w
    position_ids = (
        patch_positions[None, :, 0] * (pixtral_config.image_size // pixtral_config.patch_size)
        + patch_positions[None, :, 1]
    )
    output_pixtral_query, output_pixtral_key = transformers.models.pixtral.modeling_pixtral.apply_rotary_pos_emb(
        query, key, *pixtral_rotary(query, position_ids), unsqueeze_dim=2
    )

    fast_llm_rotary = Rotary2DConfig().get_layer(TensorDim("head_dim", head_dim))
    kwargs = {VisionKwargs.patch_positions: patch_positions, AttentionKwargs.device: "cuda"}
    fast_llm_rotary.preprocess(kwargs)
    output_fast_llm_query, output_fast_llm_key = fast_llm_rotary.forward(query, key, kwargs)

    Assert.rms_close(output_pixtral_query, output_fast_llm_query, 1e-5)
    Assert.rms_close(output_pixtral_key, output_fast_llm_key, 1e-5)
