import click
import torch
import transformers
from transformers import AutoConfig, MistralForCausalLM

from fast_llm.models.ssm.external.apriel_15b_hybrid.configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig
from fast_llm.models.ssm.external.apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import (
    AprielThinkerSSMHybridForCausalLM,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Transformers version:", transformers.__version__)


@click.command()
@click.option("--index_to_swap", type=int, required=True)
@click.option("--checkpoint", type=str, required=True)
@click.option("--output_model_path", type=str, required=True)
@click.option("--layer_type", type=str, default="m2")
def main(
    index_to_swap: int,
    checkpoint=None,
    output_model_path="/mnt/checkpoints/ssm/iterative_hybrids_15b_rkl_m2/apriel_ssm_thinker_15b_hybrid",
    layer_type="m2",
):
    print(f"index_to_swap: {index_to_swap}, checkpoint: {checkpoint}")

    layer_importance = [
        47,
        39,
        24,
        36,
        31,
        43,
        32,
        20,
        38,
        37,
        30,
        33,
        22,
        23,
        40,
        42,
        44,
        35,
        41,
        27,
        21,
        46,
        45,
        49,
        25,
        34,
        29,
        28,
        19,
        26,
        18,
        17,
        16,
        13,
        15,
        14,
        8,
        9,
        12,
        6,
        11,
        5,
        48,
        7,
        10,
        3,
        4,
        1,
        0,
    ]
    path_base = "/mnt/checkpoints/upstream/Apriel-Nemotron-15b-Thinker"
    config_base = AutoConfig.from_pretrained(path_base)
    hybrid_block_layout = ["t"] * config_base.num_hidden_layers

    for i in range(index_to_swap + 1):
        layer_idx = int(layer_importance[i])
        print(f"Swapping layer {layer_idx} to {layer_type}")
        hybrid_block_layout[layer_idx] = layer_type

    if checkpoint is None:
        print("Loading base model from thinker checkpoint")
        model_base = MistralForCausalLM.from_pretrained(path_base).to(torch.bfloat16)
        config_hybrid = AprielSSMHybridConfig(
            hybrid_block_layout=hybrid_block_layout,
            ssm_cfg={
                "d_state": 64,
                "n_v_heads": 24,
                "n_qk_heads": 24,
                "expand": 1,
                "chunk_size": 128,
                "activation": "identity",
                "bias": False,
                "d_conv": 4,
                "d_inner": 24 * 128,
            },
            **config_base.to_dict(),
        )

    else:
        print(f"Loading base model from checkpoint: {checkpoint}")
        model_base = AprielThinkerSSMHybridForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(
            torch.bfloat16
        )
        config_hybrid = AprielSSMHybridConfig(**model_base.config.to_dict())
        config_hybrid.hybrid_block_layout = hybrid_block_layout

    model_hybrid = AprielThinkerSSMHybridForCausalLM(config_hybrid)

    missing, unexpected = model_hybrid.load_state_dict(model_base.state_dict(), strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model_hybrid.to(torch.bfloat16)

    print(model_hybrid)
    model_hybrid.save_pretrained(f"{output_model_path}")


if __name__ == "__main__":
    main()
    # main(index_to_swap=1,
    #     checkpoint="/mnt/checkpoints/fast_llm_exp/slam_ssm_distill/15b-ihyb1lrklm216mil-bs768-lr0.0003-lrs0-0-0-0-sl4096_ti1000_lm2/export/apriel_ssm_thinker_hybrid/1000",
    #      layer_type="m2")
