import click
import torch
import transformers
from transformers import AutoConfig, MistralForCausalLM

from fast_llm.models.ssm.external.apriel_hybrid.modeling_ssm_hybrid_apriel import AprielSSMHybridConfig
from fast_llm.models.ssm.external.apriel_hybrid.modeling_ssm_hybrid_apriel import AprielSSMHybridModel, AprielSSMDecoderLayer, AprielSSMHybridForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Transformers version:", transformers.__version__)


@click.command()
@click.option("--index_to_swap", type=int, required=True)
@click.option("--checkpoint", type=str, required=True)
@click.option("--output_model_path", type=str, required=True)
def main(
    index_to_swap: int,
    checkpoint=None,
    output_model_path="/mnt/checkpoints/ssm/iterative_hybrids_15b/apriel_ssm_base_15b_hybrid",
    
):
    print(f"index_to_swap: {index_to_swap}, checkpoint: {checkpoint}")

    layer_importance = ['22', '25', '20', '31', '29', '46', '23', '26', '33', '24', '47', '27', '21', '41', '17', '18', '34', '42', '44', '30', '16', '8', '43', '35', '19', '38', '15', '28', '32', '45', '37', '40', '7', '36', '13', '10', '5', '39', '6', '14', '4', '12', '9', '48', '1', '3', '11', '49', '0']
    path_base = "/mnt/checkpoints/upstream/Slam-15B-Upcycled"
    config_base = AutoConfig.from_pretrained(path_base)
    hybrid_block_layout = ["t"] * config_base.num_hidden_layers
    
    for i in range(index_to_swap + 1):
        layer_idx = int(layer_importance[i])
        print(f"Swapping layer {layer_idx} to m2d")
        hybrid_block_layout[layer_idx] = "m2d"

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

    model_hybrid = AprielSSMHybridForCausalLM(config_hybrid)

    if checkpoint is None:
        print("Loading base model from instruct checkpoint")
        path_base = path_base
        model_base = MistralForCausalLM.from_pretrained(path_base).to(torch.bfloat16)
    else:
        print(f"Loading base model from checkpoint: {checkpoint}")
        path_base = checkpoint
        model_base = AprielSSMHybridForCausalLM.from_pretrained(path_base, trust_remote_code=True).to(torch.bfloat16)

    missing, unexpected = model_hybrid.load_state_dict(model_base.state_dict(), strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    model_hybrid.to(torch.bfloat16)

    print(model_hybrid)
    model_hybrid.save_pretrained(
        f"{output_model_path}"
    )


if __name__ == "__main__":
    main()