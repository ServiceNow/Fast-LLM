import pytest

from fast_llm.models.ssm.external.apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import NemotronHMamba2Mixer
from fast_llm.models.ssm.external.nemotron.config import NemotronHConfig
from fast_llm.models.ssm.external.nemotron.modeling import NemotronHMamba2Mixer as NemotronHMamba2Mixer_original


# in apriel's mamba2 mixer we do not used groups for B and C, but we have the d_xb dim, that simulates GQA
# so in order to reconstruct the original nemotron mixer, we need to set d_xb same as d_inner
@pytest.mark.parametrize(
    "apriel_ssm_config, nemotron_h_config",
    [
        (
            {
                "d_state": 16,
                "d_xb": 4096,
                "expand": 1,
                "d_conv": 4,
                "d_inner": 4096,
                "conv_bias": True,
                "bias": False,
                "head_dim": 128,  # 4096/128 = 32 heads, 1024/128 = 8 KVheads and 4 repeat groups
            },
            NemotronHConfig(
                hidden_size=4096,
                mamba_num_heads=32,
                mamba_head_dim=128,
                mamba_n_groups=32,
                mamba_d_conv=4,
                mamba_expand=1,
                ssm_state_size=16,
                use_bias=False,
                mamba_hidden_act="silu",
            ),
        )
    ],
)
def test_nemotron_h_mamba2_mixers_identical(apriel_ssm_config: dict, nemotron_h_config: dict):
    mixer_apriel = NemotronHMamba2Mixer(d_model=4096, **apriel_ssm_config)
    mixer_nemotron_h = NemotronHMamba2Mixer_original(nemotron_h_config, 0)

    for k_a, v_a in mixer_apriel.state_dict().items():
        if k_a == "dt_in_proj.weight":
            continue
        v_b = mixer_nemotron_h.state_dict()[k_a]
        if k_a == "in_proj.weight":
            assert [v_a.shape[0], v_a.shape[1]] == [v_b.shape[0] - nemotron_h_config.mamba_num_heads, v_b.shape[1]]
        else:
            assert v_a.shape == v_b.shape


if __name__ == "__main__":
    pytest.main([__file__])
