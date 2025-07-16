import click
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from fast_llm.models.ssm.external.apriel_15b_hybrid.configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig
from fast_llm.models.ssm.external.apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import (
    AprielSSMM2DecoderLayer,
    AprielThinkerSSMHybridForCausalLM,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Transformers version:", transformers.__version__)


def convert_layers(transformer, mamba_config, hybrid_block_layout, init_with_kqvo, torch_dtype):

    for layer_idx, type in enumerate(hybrid_block_layout):
        # print("Converting layer %d...", layer_idx)
        # Fetch the layer module for easier access
        layer_module = transformer.model.layers._modules[f"{layer_idx}"]
        if type == "t":
            print("Skipping transformer layer %d..." % layer_idx)
        elif type == "m2":
            print("Converting layer %d to Mamba2 with MIL init..." % layer_idx)
            # Use MambaDecoderLayer for the remaining layers
            mamba_encoder = AprielSSMM2DecoderLayer(
                mamba_config,
                layer_idx,
                device="cpu",
                dtype=torch_dtype,
            )

            mamba_encoder.mlp.load_state_dict(layer_module.mlp.state_dict())
            mamba_encoder.input_layernorm.load_state_dict(layer_module.input_layernorm.state_dict())
            mamba_encoder.post_attention_layernorm.load_state_dict(layer_module.post_attention_layernorm.state_dict())
            mamba_encoder.mixer.out_proj.load_state_dict(layer_module.self_attn.o_proj.state_dict())

            if init_with_kqvo:
                # Copy weights: [z, x, B, C, dt], x -> v, B -> k, C -> q
                mamba_encoder.mixer.in_proj.weight.data[
                    mamba_config.ssm_cfg["d_inner"] : mamba_config.ssm_cfg["d_inner"] + mamba_config.ssm_cfg["d_xb"], :
                ].copy_(layer_module.self_attn.v_proj.weight.data)
                mamba_encoder.mixer.in_proj.weight.data[
                    mamba_config.ssm_cfg["d_inner"]
                    + mamba_config.ssm_cfg["d_xb"] : mamba_config.ssm_cfg["d_inner"]
                    + 2 * mamba_config.ssm_cfg["d_xb"],
                    :,
                ].copy_(layer_module.self_attn.k_proj.weight.data)
                mamba_encoder.mixer.in_proj.weight.data[
                    mamba_config.ssm_cfg["d_inner"]
                    + 2 * mamba_config.ssm_cfg["d_xb"] : 2 * mamba_config.ssm_cfg["d_inner"]
                    + 2 * mamba_config.ssm_cfg["d_xb"],
                    :,
                ].copy_(layer_module.self_attn.q_proj.weight.data)

                print("Init Mamba using Attention")

            transformer.model.layers[layer_idx] = mamba_encoder

        elif type == "m2d":
            raise NotImplementedError("Discrete Mamba2 not implemented")
        else:
            raise ValueError(f"Invalid layer type: {type}")


@click.command()
@click.option("--index_to_swap", type=int, required=True)
@click.option("--checkpoint", type=str, required=True)
@click.option("--output_model_path", type=str, required=True)
@click.option("--layer_type", type=str, default="m2")
@click.option("--mil_init", type=bool, default=True)
def main(
    index_to_swap: int,
    checkpoint=None,
    output_model_path="/mnt/checkpoints/ssm/iterative_hybrids_15b_rkl_m2/apriel_ssm_thinker_15b_hybrid",
    layer_type="m2",
    mil_init=True,
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

    transformer = AutoModelForCausalLM.from_pretrained(path_base)
    model_hybrid_prev = AprielThinkerSSMHybridForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(
        torch.bfloat16
    )
    config_hybrid = AprielSSMHybridConfig(**model_hybrid_prev.config.to_dict())
    config_hybrid.hybrid_block_layout = hybrid_block_layout
    convert_layers(transformer, config_hybrid, hybrid_block_layout, mil_init, torch.bfloat16)

    missing, unexpected = transformer.load_state_dict(
        model_hybrid_prev.state_dict(), strict=False
    )  # will not load the newly innitialized layer (will stay MIL), but will overwrite previous layers
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    transformer.to(torch.bfloat16)
    model_hybrid_prev = None
    print(transformer)
    model_hybrid = AprielThinkerSSMHybridForCausalLM(config_hybrid)
    missing, unexpected = model_hybrid.load_state_dict(transformer.state_dict())
    assert len(missing) == 0, "Missing keys: " + str(missing)
    assert len(unexpected) == 0, "Unexpected keys: " + str(unexpected)

    model_hybrid.save_pretrained(f"{output_model_path}")
    # config_hybrid.save_pretrained(f"{output_model_path}")


if __name__ == "__main__":
    main()
    # main(
    #     index_to_swap=1,
    #     checkpoint="/mnt/checkpoints/fast_llm_exp/slam_ssm_distill/15b-ihyb1lrklm216mil-bs768-lr0.0003-lrs0-0-0-0-sl4096_ti1000_lm2/export/apriel_ssm_thinker_hybrid/1000",
    #     layer_type="m2",
    # )
