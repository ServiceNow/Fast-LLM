import pytest
import torch
import transformers

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, ProportionalRotaryConfig, Rotary2DConfig
from fast_llm.layers.vision.config import VisionKwargs
from fast_llm.utils import Assert


def test_rotary_2d(testing_device):
    """
    Compare Fast-LLM's implementation of 2d rotary embeddings with Pixtral.
    """
    head_dim = 16
    num_heads = 8

    patch_positions = torch.tensor(
        [[h, w] for h in range(4) for w in range(4)],
        dtype=torch.int64,
        device=testing_device,
    )
    query = torch.empty(
        2, len(patch_positions), num_heads, head_dim, dtype=torch.float32, device=testing_device
    ).normal_()
    key = torch.empty_like(query).normal_()
    value = torch.empty_like(query).normal_()

    pixtral_config = transformers.PixtralVisionConfig(hidden_size=head_dim * num_heads, num_attention_heads=num_heads)
    pixtral_rotary = transformers.models.pixtral.modeling_pixtral.PixtralRotaryEmbedding(pixtral_config).to(
        testing_device
    )
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
    kwargs = {VisionKwargs.patch_positions: patch_positions, AttentionKwargs.device: testing_device}
    fast_llm_rotary.preprocess(kwargs)
    output_fast_llm_query, output_fast_llm_key_value = fast_llm_rotary.forward(
        query, torch.cat([key, value], dim=-2), kwargs
    )
    output_fast_llm_key, output_fast_llm_value_ = output_fast_llm_key_value.chunk(2, dim=-2)
    Assert.rms_close(output_pixtral_query, output_fast_llm_query, 1e-5)
    Assert.rms_close(output_pixtral_key, output_fast_llm_key, 1e-5)
    Assert.all_equal(output_fast_llm_value_, value)


@pytest.mark.parametrize("partial_rotary_factor", [0.25, 0.5, 1.0])
def test_proportional_rotary(testing_device, partial_rotary_factor):
    """
    Verify ProportionalRotary rotates only the first partial_rotary_factor * head_size
    dimensions and passes the rest through unchanged, matching DefaultRotary on the rotated slice.
    """
    head_size, heads, head_groups, seq_len = 32, 4, 2, 20
    rotary_dims = int(head_size * partial_rotary_factor)
    theta = 10000.0

    query = torch.randn(seq_len, heads, head_size, dtype=torch.float32, device=testing_device)
    key_value = torch.randn(seq_len, 2 * head_groups, head_size, dtype=torch.float32, device=testing_device)
    k, v = key_value.chunk(2, dim=-2)

    def _make_kwargs(sequence_length):
        return {
            AttentionKwargs.sequence_length: sequence_length,
            AttentionKwargs.sequence_k_dim: TensorDim("seq_k", sequence_length),
            AttentionKwargs.token_dim: TensorDim("token", sequence_length),
            AttentionKwargs.device: testing_device,
        }

    # Proportional rotary under test
    proportional = ProportionalRotaryConfig(theta=theta, partial_rotary_factor=partial_rotary_factor).get_layer(
        TensorDim("head_size", head_size)
    )
    prop_kwargs = _make_kwargs(seq_len)
    proportional.preprocess(prop_kwargs)
    # triton_rotary_ modifies tensors in-place; for factor=1.0 the full-head-size slice
    # is already contiguous so .contiguous() returns the same storage. Clone to keep
    # the original query available for the reference comparison below.
    out_q, out_kv, _ = proportional.forward_only(query.clone(), key_value, prop_kwargs)
    out_k, out_v = out_kv.chunk(2, dim=-2)

    # Reference: DefaultRotary on the first rotary_dims only
    reference = DefaultRotaryConfig(theta=theta).get_layer(TensorDim("head_size_partial", rotary_dims))
    ref_kwargs = _make_kwargs(seq_len)
    reference.preprocess(ref_kwargs)
    ref_q_rot, ref_kv_rot, _ = reference.forward_only(
        query[..., :rotary_dims].contiguous(),
        torch.cat([k[..., :rotary_dims].contiguous(), v[..., :rotary_dims].contiguous()], dim=-2),
        ref_kwargs,
    )
    ref_k_rot, _ = ref_kv_rot.chunk(2, dim=-2)

    # Rotated portion matches reference DefaultRotary on the slice
    Assert.rms_close(out_q[..., :rotary_dims], ref_q_rot, 1e-5)
    Assert.rms_close(out_k[..., :rotary_dims], ref_k_rot, 1e-5)

    # Non-rotated portion is unchanged
    Assert.all_equal(out_q[..., rotary_dims:], query[..., rotary_dims:])
    Assert.all_equal(out_k[..., rotary_dims:], k[..., rotary_dims:])

    # Value always passes through unchanged
    Assert.all_equal(out_v, v)
