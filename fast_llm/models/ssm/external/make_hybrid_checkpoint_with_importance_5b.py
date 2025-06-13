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
    output_model_path="/mnt/checkpoints/ssm/iterative_hybrids_5b/apriel_ssm_instruct5b_hybrid_only_new_layer_train",
    
):
    print(f"index_to_swap: {index_to_swap}, checkpoint: {checkpoint}")

    layer_importance = ['22', '24', '19', '27', '20', '5', '4', '9', '23', '7', '8', '6', '2', '26', '11', '14', '15', '3', '1', '13', '16', '10', '12', '17', '25', '18', '0']

    path_instruct = "/mnt/checkpoints/upstream/Apriel-5B-Instruct-llamafied/"
    config_instruct = AutoConfig.from_pretrained(path_instruct)
    hybrid_block_layout = ["t"] * config_instruct.num_hidden_layers
    
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
        **config_instruct.to_dict(),
    )

    model_hybrid = AprielSSMHybridForCausalLM(config_hybrid)

    if checkpoint is None:
        print("Loading base model from instruct checkpoint")
        path_base = path_instruct
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
        f"{output_model_path}/ihyb{index_to_swap+1}l24h/export/apriel_ssm_instruct5b_hybrid/400"
    )


if __name__ == "__main__":
    main()