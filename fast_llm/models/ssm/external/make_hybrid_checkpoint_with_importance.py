import gc

import click
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import transformers
from transformers import MistralForCausalLM
from fast_llm.models.ssm.external.apriel_15b_hybrid.configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig
from fast_llm.models.ssm.external.apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import AprielSSMHybridForCausalLM

# from apriel_15b_hybrid.configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig
# from apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import AprielSSMHybridForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Transformers version:", transformers.__version__)

@click.command()
@click.option("--index_to_swap", type=int, required=True)
@click.option("--checkpoint", type=str, required=True)
def main(index_to_swap: int, checkpoint=None):
    print(f"index_to_swap: {index_to_swap}, checkpoint: {checkpoint}")
    
    layer_importance = [
        47, 39, 24, 20, 38, 30, 43, 36, 31, 37, 49, 32, 33, 35, 44, 45, 42, 22, 41, 40,
        23, 21, 46, 29, 34, 27, 25, 28, 19, 26, 18, 17, 16, 13, 15, 14, 8, 9, 12, 6, 11,
        48, 5, 10, 7, 3, 4, 1, 0
    ]
    
    path_thinker = "/mnt/checkpoints/upstream/Apriel-Nemotron-15b-Thinker"
    config_thinker = AutoConfig.from_pretrained(path_thinker)
    hybrid_block_layout = ["t"] * config_thinker.num_hidden_layers
    
    for i in range(index_to_swap + 1):
        hybrid_block_layout[layer_importance[i]] = "m2d"
    
    # checkpoint_model = AprielSSMHybridForCausalLM.from_pretrained(path_thinker)
    # hybrid_block_layout = checkpoint_model.config.hyb_block_layout # ["t"] * config_thinker.num_hidden_layers
    # hybrid_block_layout[layer_importance[layer_index_to_swap]] = "m2d"

    config_hybrid = AprielSSMHybridConfig(
        **config_thinker.to_dict(),
        hybrid_block_layout=hybrid_block_layout,
        ssm_cfg = {
                "d_state": 64,
                "n_v_heads": 32,
                "n_qk_heads": 32,
                "expand": 1,
                "chunk_size": 128,
                "activation": "identity",
                "bias": False,
                "d_conv": 4,
                "d_inner": 32 * 128
            }
    )
    
    model_hybrid = AprielSSMHybridForCausalLM(config_hybrid)
    
    if checkpoint is None:
        path_base = path_thinker
        model_base = MistralForCausalLM.from_pretrained(path_base).to(torch.bfloat16)
    else:
        path_base = checkpoint
        model_base = AprielSSMHybridForCausalLM.from_pretrained(path_base, trust_remote_code=True).to(torch.bfloat16)

    model_hybrid.load_state_dict(model_base.state_dict(), strict=False)
    model_hybrid.save_pretrained(f"/mnt/checkpoints/ssm/iterative_hybrids/apriel_ssm_thinker15b_hybrid_{index_to_swap+1}ssm_leastimportant_32h_init_rand")

    # checkpoint = "ServiceNow-AI/Apriel-Nemotron-15b-Thinker"
    # config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)

    # hybrid_block_layout = ["t"] * config.num_hidden_layers
    # if identity_index >= 0:
    #     hybrid_block_layout[identity_index] = "i"

    # hybrdif_apriel_config = AprielSSMHybridConfig(**config.to_dict(), hybrid_block_layout=hybrid_block_layout)
    # hybrid_apriel_model = AprielSSMHybridForCausalLM(hybrdif_apriel_config)
    # hybrid_apriel_model.to(dtype=torch.bfloat16).to(device)

    # apriel_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True)
    # apriel_state_dict = apriel_model.state_dict()
    # hybrid_apriel_model.load_state_dict(apriel_state_dict, strict=False)

    # hybrid_apriel_model.save_pretrained(save_dir, save_config=True)
    # torch.cuda.empty_cache()
    # del hybrid_apriel_model
    # del apriel_model
    # del apriel_state_dict
    # gc.collect()


if __name__ == "__main__":
    main()
