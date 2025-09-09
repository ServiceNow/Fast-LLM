import gc
import json
import shutil

import click
import torch
from transformers import AutoModelForVision2Seq

from fast_llm.models.ssm.external.apriel_15b_hybrid import modeling_ssm_hybrid_apriel15b
from fast_llm.models.ssm.external.llava_hybrid import configuration_llava_hybrid, modeling_llava_hybrid
from fast_llm.models.ssm.external.llava_hybrid.configuration_llava_hybrid import LlavaHybridConfig
from fast_llm.models.ssm.external.llava_hybrid.modeling_llava_hybrid import LlavaHybridForConditionalGeneration
from fast_llm.models.ssm.external.make_hybrid_checkpoint import convert_layers

device = "cuda" if torch.cuda.is_available() else "cpu"

dstate = 16
expand = 1
# Calculate derived dimensions for the Mamba1 configuration
# d_model = config_base.text_config.hidden_size
d_inner = 4096  # hard code to match thinker #expand * d_model
d_xb = 1024  # hard code to match thinker #config_thinker.num_key_value_heads * (config_thinker.hidden_size // config_thinker.num_attention_heads)


def make_hybrid_llava_config(transformer):
    config_dict = transformer.config.to_dict()
    config_dict["text_config"]["hybrid_block_layout"] = ["t"] * transformer.config.text_config.num_hidden_layers
    config_dict["text_config"]["model_type"] = "apriel_ssm_thinker_hybrid"
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
@click.option("--base_checkpoint", type=str, required=False, default="ServiceNow-AI/Apriel-Nemotron-15b-Thinker")
@click.option("--m2_indices", type=int, multiple=True, required=True)
@click.option("--hybrid_checkpoint", type=str, required=True)
@click.option("--save_dir", type=str, required=True)
@click.option(
    "--tokenizer_dir", type=str, required=False, default="/mnt/plato/checkpoints/upstream/Mistral-Nemo-Base-2407/"
)
def main(base_checkpoint: str, m2_indices: list[int], hybrid_checkpoint: str, save_dir: str, tokenizer_dir: str):
    """
    base_checkpoint: path to base transformer-model (teacher model)
    m2_indices: indices of layers to convert to mamba layers with MiL init
    hybrid_checkpoint: path to hybrid model (student model). Can be a hybrid with only transformer layers for the first distillation run.
    save_dir: directory to save the converted model.
    tokenizer_dir: directory containing tokenizer files to copy over to save_dir.
    """
    m2_indexes = list(m2_indices)  # convert tuple -> list
    transformer = AutoModelForVision2Seq.from_pretrained(base_checkpoint, trust_remote_code=True)
    if hybrid_checkpoint == "none":
        print("No hybrid checkpoint provided, creating new config from base model.")
        hybrid_config = make_hybrid_llava_config(transformer)

        hybrid_llava_model = None
    else:
        hybrid_config = LlavaHybridConfig.from_pretrained(hybrid_checkpoint)
        # Load existing SSM layers
        hybrid_llava_model = AutoModelForVision2Seq.from_pretrained(
            hybrid_checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

    hybrid_block_layout = hybrid_config.text_config.hybrid_block_layout
    for m2_index in m2_indexes:
        hybrid_block_layout[m2_index] = "m2"
    print(hybrid_block_layout)

    # MiL init
    convert_layers(
        transformer.model.language_model.config,
        transformer.model.language_model,
        hybrid_config.text_config,
        hybrid_block_layout,
        init_with_kqvo=True,
        torch_dtype=torch.bfloat16,
    )
    hybrid_config.text_config.ssm_cfg["activation"] = "silu"

    # Load existing SSM layers
    if hybrid_checkpoint != "none":
        llava_state_dict = hybrid_llava_model.state_dict()
        missing, unexpected = transformer.load_state_dict(llava_state_dict, strict=False)
        for m2_index in m2_indexes:
            assert f"model.layers.{m2_index}.mixer.A_log" in missing
            assert f"model.layers.{m2_index}.self_attn.q_proj.weight" in unexpected
        print("MISSING", missing)
        print("UNEXPECTED", unexpected)

    # Save state-dict
    transformer.save_pretrained(save_dir)
    # Save new config
    hybrid_config.save_pretrained(save_dir)

    # Copy modeling and tokenizer files
    modeling_files = [
        configuration_llava_hybrid.__file__,
        modeling_llava_hybrid.__file__,
        modeling_ssm_hybrid_apriel15b.__file__,
    ]
    tokenizer_files = [
        f"{tokenizer_dir}/tokenizer.json",
        f"{tokenizer_dir}/tokenizer_config.json",
        f"{tokenizer_dir}/generation_config.json",
        f"{tokenizer_dir}/special_tokens_map.json",
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

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
