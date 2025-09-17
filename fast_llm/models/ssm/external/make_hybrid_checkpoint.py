import gc

import click
import torch
from transformers import AutoModelForCausalLM

from fast_llm.models.ssm.external.apriel_15b_hybrid.configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig
from fast_llm.models.ssm.external.apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import (
    AprielSSMM2DecoderLayer,
    AprielThinkerSSMHybridForCausalLM,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

dstate = 16
expand = 1
# Calculate derived dimensions for the Mamba1 configuration
# d_model = config_base.text_config.hidden_size
d_inner = 4096  # hard code to match thinker #expand * d_model
d_xb = 1024  # hard code to match thinker #config_thinker.num_key_value_heads * (config_thinker.hidden_size // config_thinker.num_attention_heads)


def convert_layers(
    transformer_config,
    transformer_model,
    mamba_config,
    hybrid_block_layout,
    init_with_kqvo,
    torch_dtype=torch.bfloat16,
):
    config = transformer_config
    embed_dim = config.hidden_size
    num_heads = config.num_attention_heads
    num_heads_kv = config.num_key_value_heads
    head_dim = embed_dim // num_heads
    head_dim * num_heads
    head_dim * num_heads_kv

    for layer_idx, type in enumerate(hybrid_block_layout):
        print("Converting layer %d...", layer_idx)
        # Fetch the layer module for easier access
        layer_module = transformer_model.layers._modules[f"{layer_idx}"]
        if type == "t":
            print("Skipping transformer layer %d..." % layer_idx)
        elif type == "m2":
            print("Converting layer %d..." % layer_idx)
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

            transformer_model.layers[layer_idx] = mamba_encoder

        else:
            raise ValueError(f"Invalid layer type: {type}")


def make_hybrid_config(transformer):
    config_dict = transformer.config.to_dict()
    config_dict["hybrid_block_layout"] = ["t"] * transformer.config.num_hidden_layers
    config_dict["model_type"] = "apriel_ssm_thinker_hybrid"
    config_dict["ssm_cfg"] = {
        "activation": "silu",
        "d_state": dstate,
        "d_xb": d_xb,
        "expand": expand,
        "d_conv": 4,
        "d_inner": d_inner,
        "conv_bias": True,
        "bias": False,
    }
    hybrid_config = AprielSSMHybridConfig.from_dict(**config_dict)
    return hybrid_config


@click.command()
@click.option(
    "--base_checkpoint", type=str, required=False, default="/mnt/checkpoints/upstream/Apriel-Nemotron-15b-Thinker"
)
@click.option("--m2_indices", type=int, multiple=True, required=True)
@click.option("--hybrid_checkpoint", type=str, required=True)
@click.option("--save_dir", type=str, required=True)
def main(base_checkpoint: str, m2_indices: list, hybrid_checkpoint: str, save_dir: str):
    """
    base_checkpoint: path to base transformer-model (teacher model)
    m2_indices: indices of layers to convert to mamba layers with MiL init
    hybrid_checkpoint: path to hybrid model (student model).
    save_dir: directory to save the converted model.

    TODO: base_checkpoint can actually be a hybrid. Rename transformer variable to a better name
    """
    m2_indices = list(m2_indices)  # convert tuple -> list
    transformer = AutoModelForCausalLM.from_pretrained(base_checkpoint, trust_remote_code=True)
    if hybrid_checkpoint == "none":
        print("No hybrid checkpoint provided, creating new config from base model.")
        hybrid_config = make_hybrid_config(transformer)
    else:
        hybrid_config = AprielSSMHybridConfig.from_pretrained(hybrid_checkpoint)

    hybrid_block_layout = hybrid_config.hybrid_block_layout
    for m2_index in m2_indices:
        hybrid_block_layout[m2_index] = "m2"
    print(hybrid_block_layout)

    convert_layers(
        transformer.config,
        transformer.model,
        hybrid_config,
        hybrid_block_layout,
        init_with_kqvo=True,
        torch_dtype=torch.bfloat16,
    )
    hybrid_config.ssm_cfg["activation"] = "silu"

    # load all existing  ssm layers
    if hybrid_checkpoint != "none":
        hybrid_model = AprielThinkerSSMHybridForCausalLM.from_pretrained(hybrid_checkpoint)
        state_dict = hybrid_model.state_dict()
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        for m2_index in m2_indices:
            assert f"model.layers.{m2_index}.mixer.A_log" in missing
            assert f"model.layers.{m2_index}.self_attn.q_proj.weight" in unexpected
        print("MISSING", missing)
        print("UNEXPECTED", unexpected)

    # Save state-dict
    transformer.save_pretrained(save_dir)

    hybrid_config.save_pretrained(save_dir)

    gc.collect()


if __name__ == "__main__":
    main()
