import gc
import json
import os
import shutil

import click
import torch
from transformers import AutoConfig

from fast_llm.models.ssm.external.apriel_15b_hybrid import (
    configuration_ssm_hybrid_apriel15b,
    modeling_ssm_hybrid_apriel15b,
)
from fast_llm.models.ssm.external.llava_hybrid import configuration_llava_hybrid, modeling_llava_hybrid
from fast_llm.models.ssm.external.llava_hybrid.configuration_llava_hybrid import LlavaHybridConfig
from fast_llm.models.ssm.external.llava_hybrid.modeling_llava_hybrid import LlavaHybridForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
# swa_size = 2048
dstate = 16
expand = 1
# Calculate derived dimensions for the Mamba1 configuration
# d_model = config_base.text_config.hidden_size
d_inner = 4096  # hard code to match thinker #expand * d_model
d_xb = 1024  # hard code to match thinker #config_thinker.num_key_value_heads * (config_thinker.hidden_size // config_thinker.num_attention_heads)


def make_hybrid_llava_config(transformer_config, swa_size):
    config_dict = transformer_config.to_dict()
    config_dict["text_config"]["model_type"] = "apriel_ssm_thinker_hybrid"
    if "swa" in transformer_config.text_config.hybrid_block_layout:
        config_dict["text_config"]["sliding_window"] = swa_size
    if "dtype" not in config_dict["text_config"] or config_dict["text_config"]["dtype"] is None:
        config_dict["text_config"]["dtype"] = config_dict["dtype"]
    config_dict["text_config"]["ssm_cfg"] = {
        "activation": "silu",
        "d_state": dstate,
        "d_xb": d_xb,
        # "d_model": d_model, # will be set automatically
        "expand": expand,
        "d_conv": 4,
        "d_inner": d_inner,  # will be same as d_model * expand,
        "conv_bias": True,
        "bias": False,
    }
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_llava_hybrid.LlavaHybridConfig",
        "AutoModel": "modeling_llava_hybrid.LlavaHybridModel",
        "AutoModelForCausalLM": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
        "AutoModelForVision2Seq": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
    }
    config_dict["text_config"]["auto_map"] = {
        "AutoConfig": "configuration_ssm_hybrid_apriel15b.AprielSSMHybridConfig",
        "AutoModel": "modeling_ssm_hybrid_apriel15b.AprielThinkerSSMHybridModel",
        "AutoModelForCausalLM": "modeling_ssm_hybrid_apriel15b.AprielThinkerSSMHybridForCausalLM",
    }
    llava_hybrid_config = LlavaHybridConfig(**config_dict)
    return llava_hybrid_config


def make_hybrid_llava_model(transformer, llava_hybrid_config):
    """
    Create a LlavaHybridForConditionalGeneration model with the same configuration as the given transformer model.
    """
    llava_hybrid_model = LlavaHybridForConditionalGeneration(llava_hybrid_config)
    # llava_hybrid_model.to(dtype=torch.bfloat16).to(device)
    llava_hybrid_model.load_state_dict(transformer.state_dict(), strict=False)
    return llava_hybrid_model


@click.command()
@click.option(
    "--base_checkpoint", type=str, required=False, default="/mnt/checkpoints/upstream/Apriel-1.5-15b-Thinker"
)
@click.option("--n_swa", type=int, required=False, default=0)
@click.option("--n_gdn", type=int, default=1, required=False)
@click.option("--n_kl", type=int, default=0, required=False)
@click.option(
    "--save_dir",
    type=str,
    required=False,
    default="/mnt/checkpoints/ssm/vllm_checkpoints/apriel_hybrid_throughput_checkpoints/checkpoints_gdn_swa_2048/test",
)
@click.option("--skip_if_exists", is_flag=True, default=False)
@click.option("--tokenizer_dir", type=str, required=False, default="/mnt/checkpoints/upstream/Apriel-1.5-15b-Thinker")
@click.option("--swa_size", type=int, required=False, default=2048)
def main(
    base_checkpoint: str,
    n_swa: int,
    n_gdn: int,
    n_kl: int,
    save_dir: str,
    skip_if_exists: bool,
    tokenizer_dir: str,
    swa_size: int,
):
    """
    base_checkpoint: path to base transformer-model (teacher model)
    m2_indices: indices of layers to convert to mamba layers with MiL init
    hybrid_checkpoint: path to hybrid model (student model). Can be a hybrid with only transformer layers for the first distillation run.
    save_dir: directory to save the converted model.
    tokenizer_dir: directory containing tokenizer files to copy over to save_dir.
    """
    if skip_if_exists and os.path.exists(save_dir):
        print(f"Checkpoint {save_dir} already exists, skipping...")
        return
    if n_swa + n_gdn + n_kl > 48:
        raise ValueError("n_swa + n_gdn + n_kl exceeds total number of layers (48)")

    base_config = AutoConfig.from_pretrained(base_checkpoint, trust_remote_code=True)

    hybrid_block_layout = ["t"] * base_config.text_config.num_hidden_layers
    assert (
        n_swa + n_gdn + n_kl <= base_config.text_config.num_hidden_layers
    ), "n_swa + n_gdn + n_kl exceeds total number of layers"

    for swa_idx in range(n_swa):
        hybrid_block_layout[swa_idx] = "swa"
    for gdn_idx in range(n_gdn):
        hybrid_block_layout[gdn_idx + n_swa] = "gdn"
    for kl_idx in range(n_kl):
        hybrid_block_layout[kl_idx + n_swa + n_gdn] = "kl"

    setattr(base_config.text_config, "hybrid_block_layout", hybrid_block_layout)
    hybrid_config = make_hybrid_llava_config(base_config, swa_size)

    print(hybrid_config.text_config.hybrid_block_layout)

    hybrid_config.text_config.ssm_cfg["activation"] = "silu"
    llava_hybrid_model = LlavaHybridForConditionalGeneration(hybrid_config)

    # Save state-dict
    llava_hybrid_model.save_pretrained(save_dir)  # here dtype is set to float32 for some reason
    # Save new config
    hybrid_config.save_pretrained(save_dir)

    # Copy modeling and tokenizer files
    modeling_files = [
        configuration_ssm_hybrid_apriel15b.__file__,
        configuration_llava_hybrid.__file__,
        modeling_llava_hybrid.__file__,
        modeling_ssm_hybrid_apriel15b.__file__,
    ]
    tokenizer_files = [
        f"{tokenizer_dir}/tokenizer.json",
        f"{tokenizer_dir}/tokenizer_config.json",
        f"{tokenizer_dir}/generation_config.json",
        f"{tokenizer_dir}/special_tokens_map.json",
        f"{tokenizer_dir}/preprocessor_config.json",
    ]
    for f in modeling_files + tokenizer_files:
        shutil.copy(f, save_dir)

    # Update config with auto_maps
    config_file = f"{save_dir}/config.json"
    with open(config_file) as f:
        dumped_config = json.load(f)

    dumped_config["auto_map"] = {
        "AutoConfig": "configuration_llava_hybrid.LlavaHybridConfig",
        "AutoModel": "modeling_llava_hybrid.LlavaHybridModel",
        "AutoModelForVision2Seq": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
        "AutoModelForCausalLM": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
    }
    dumped_config["text_config"]["auto_map"] = {
        "AutoConfig": "configuration_ssm_hybrid_apriel15b.AprielSSMHybridConfig",
        "AutoModel": "modeling_ssm_hybrid_apriel15b.AprielThinkerSSMHybridModel",
        "AutoModelForCausalLM": "modeling_ssm_hybrid_apriel15b.AprielThinkerSSMHybridForCausalLM",
    }
    dumped_config["architectures"] = ["LlavaHybridForConditionalGeneration"]
    dumped_config["text_config"]["architectures"] = ["AprielThinkerSSMHybridForCausalLM"]
    with open(config_file, "w") as f:
        json.dump(dumped_config, f, indent=2)

    print(f"Done to {save_dir}")

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
