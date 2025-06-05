import enum
import logging
import pathlib

import yaml
from transformers import AutoModelForCausalLM

from fast_llm.engine.config_utils.runnable import RunnableConfig

from fast_llm.config import Config, Field, config_class  # isort:skip

logger = logging.getLogger(__name__)


class PredictionHeadInitMethod(str, enum.Enum):
    from_existing = "from_existing"
    # from_scratch = "from_scratch"


@config_class()
class AddPredictionHeadsConfig(RunnableConfig):
    hf_checkpoint: pathlib.Path = Field()
    output_dir: pathlib.Path = Field()
    num_prediction_heads: int = Field()
    prediction_head_init_method: PredictionHeadInitMethod = Field()
    prediction_head_init_std: float = Field(default=0.0)

    def _validate(self):
        super()._validate()
        assert self.prediction_head_init_method == PredictionHeadInitMethod.from_existing

    def run(self):
        logger.info(f"Loading {self.hf_checkpoint}...")
        model = AutoModelForCausalLM.from_pretrained(self.hf_checkpoint)
        assert model.config.model_type in [
            "llama",
            "mistral",
            "apriel",
        ], f"Model type {model.config.model_type} is not supported"
        # We convert the models to MTP-Llama. It does not support sliding window.
        if model.config.model_type == "mistral":
            assert model.config.sliding_window is None
            model.config.mlp_bias = False
        state_dict = model.state_dict()

        logger.info(f"Adding Prediction Heads to {self.hf_checkpoint}...")

        # New prediction-heads' transformer layers
        hf_mtp_head_base_name = "model.mtp_heads"
        # Last layer is the first head
        last_layer_base_name = f"model.layers.{model.config.num_hidden_layers - 1}"
        for i in range(self.num_prediction_heads - 1):
            for w in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
                state_dict[f"{hf_mtp_head_base_name}.{i}.{w}.weight"] = state_dict[
                    f"{last_layer_base_name}.{w}.weight"
                ].clone()
                # Llama: no bias in attention
                assert f"{last_layer_base_name}.{w}.bias" not in state_dict, "Bias found in state_dict"
            for w in ["input_layernorm", "post_attention_layernorm"]:
                # RMS norm: no bias
                state_dict[f"{hf_mtp_head_base_name}.{i}.{w}.weight"] = state_dict[
                    f"{last_layer_base_name}.{w}.weight"
                ].clone()
                assert f"{last_layer_base_name}.{w}.bias" not in state_dict, "Bias found in state_dict"
            for w in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                state_dict[f"{hf_mtp_head_base_name}.{i}.{w}.weight"] = state_dict[
                    f"{last_layer_base_name}.{w}.weight"
                ].clone()
                if model.config.mlp_bias:
                    state_dict[f"{hf_mtp_head_base_name}.{i}.{w}.bias"] = state_dict[
                        f"{last_layer_base_name}.{w}.bias"
                    ].clone()
                else:
                    assert f"{last_layer_base_name}.{w}.bias" not in state_dict, "Bias found in state_dict"

        # Layer norms
        hf_mtp_norm_base_name = "model.mtp_norms"
        state_dict[f"{hf_mtp_norm_base_name}.0.weight"] = state_dict.pop(f"model.norm.weight")
        assert f"model.norm.bias" not in state_dict, "Bias found in state_dict"
        for i in range(1, self.num_prediction_heads):
            state_dict[f"{hf_mtp_norm_base_name}.{i}.weight"] = state_dict[f"{hf_mtp_norm_base_name}.0.weight"].clone()

        # Adjust config
        model.config.prediction_heads = self.num_prediction_heads
        model.config.auto_map = {
            "AutoConfig": "configuration_mtp_llama.MTPLlamaConfig",
            "AutoModel": "modeling_mtp_llama.MTPLlamaModel",
            "AutoModelForCausalLM": "modeling_mtp_llama.MTPLlamaForCausalLM",
        }
        # model.config.architectures = ["MTPLlamaForCausalLM"]

        # Save model
        logger.info(f"Saving model to {self.output_dir}...")
        model.save_pretrained(self.output_dir, state_dict=state_dict)
        logger.warning(
            f"WARNING: Model architecture needs to be updated manually to MTPLlamaForCausalLM in {self.output_dir}/config.json"
        )
        logger.warning(
            f"WARNING: Model-type needs to be updated manually to mtp_llama in {self.output_dir}/config.json"
        )

        # Save surgery config as yaml
        yaml.safe_dump(self.to_serialized(), (self.output_dir / "surgery_config.yaml").open("w"))
        logger.info("Done!")


if __name__ == "__main__":
    AddPredictionHeadsConfig.parse_and_run()
