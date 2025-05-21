import gc

import click
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from fast_llm.models.ssm.external.apriel_15b_hybrid.configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig
from fast_llm.models.ssm.external.apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import AprielSSMHybridForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("--identity_index", type=int, required=True)
@click.option("--save_dir", type=str, required=True)
def main(identity_index: int, save_dir: str):
    checkpoint = "ServiceNow-AI/Apriel-Nemotron-15b-Thinker"
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)

    hybrid_block_layout = ["t"] * config.num_hidden_layers
    if identity_index >= 0:
        hybrid_block_layout[identity_index] = "i"

    hybrdif_apriel_config = AprielSSMHybridConfig(**config.to_dict(), hybrid_block_layout=hybrid_block_layout)
    hybrid_apriel_model = AprielSSMHybridForCausalLM(hybrdif_apriel_config)
    hybrid_apriel_model.to(dtype=torch.bfloat16).to(device)

    apriel_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True)
    apriel_state_dict = apriel_model.state_dict()
    hybrid_apriel_model.load_state_dict(apriel_state_dict, strict=False)

    hybrid_apriel_model.save_pretrained(save_dir, save_config=True)
    torch.cuda.empty_cache()
    del hybrid_apriel_model
    del apriel_model
    del apriel_state_dict
    gc.collect()


if __name__ == "__main__":
    main()
