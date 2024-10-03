import enum
import logging
import os
import pathlib

import torch
import yaml
from transformers import AutoModelForCausalLM

from fast_llm.engine.config_utils.logging import configure_logging

try:
    import hf_transfer  # type: ignore[no-redef]
except ImportError as e:
    raise ImportError("Please install hf_transfer to use this script") from e

try:
    # must be set before importing huggingface_hub and fast_llm
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    import huggingface_hub as hf_hub
    from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER

    assert HF_HUB_ENABLE_HF_TRANSFER, "hf_transfer is not enabled"
    hf_hub.logging.set_verbosity_debug()
except ImportError as e:
    raise ImportError("Please install huggingface_hub to use this script") from e

from fast_llm.config import Config, Field, config_class  # isort:skip

logger = logging.getLogger(__name__)


class ExpertInitMethod(str, enum.Enum):
    from_existing = "from_existing"
    from_scratch = "from_scratch"


class RouterInitMethod(str, enum.Enum):
    from_existing = "from_existing"
    from_scratch = "from_scratch"


@config_class()
class AddExpertsConfig(Config):
    hf_checkpoint: pathlib.Path = Field()
    output_dir: pathlib.Path = Field()
    num_new_experts: int = Field()
    expert_init_method: ExpertInitMethod = Field()
    router_init_method: RouterInitMethod = Field()
    expert_init_std: float = Field(default=0.0)
    router_init_std: float = Field(default=0.0)

    def _validate(self):
        super()._validate()


def add_experts(config: AddExpertsConfig):
    logger.info(f"Loading {config.hf_checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(config.hf_checkpoint)
    state_dict = model.state_dict()

    logger.info(f"Adding experts to {config.hf_checkpoint}...")
    # Add experts to the model
    for layer_idx in range(model.config.num_hidden_layers):
        hf_moe_base_name = f"model.layers.{layer_idx}.block_sparse_moe"
        for expert_idx in range(config.num_new_experts):
            new_expert_idx = model.config.num_local_experts + expert_idx
            for w in ["w1", "w2", "w3"]:
                noise = (
                    torch.randn_like(state_dict[f"{hf_moe_base_name}.experts.0.{w}.weight"]) * config.expert_init_std
                )
                state_dict[f"{hf_moe_base_name}.experts.{new_expert_idx}.{w}.weight"] = (
                    state_dict[f"{hf_moe_base_name}.experts.{expert_idx}.{w}.weight"] + noise
                    if config.expert_init_method == ExpertInitMethod.from_existing
                    else noise
                )

        # New router weights
        router_weight_name = f"{hf_moe_base_name}.gate.weight"
        assert state_dict[router_weight_name].shape == (model.config.num_local_experts, model.config.hidden_size)
        new_router_weights = state_dict[router_weight_name][: config.num_new_experts].clone()
        noise = torch.randn_like(new_router_weights) * config.router_init_std
        new_router_weights = (
            new_router_weights + noise if config.router_init_method == RouterInitMethod.from_existing else noise
        )
        state_dict[router_weight_name] = torch.cat([state_dict[router_weight_name], new_router_weights], dim=0)
        assert state_dict[router_weight_name].shape == (
            model.config.num_local_experts + config.num_new_experts,
            model.config.hidden_size,
        )

    # Adjust config
    model.config.num_local_experts += config.num_new_experts

    # Save model
    logger.info(f"Saving model to {config.output_dir}...")
    model.save_pretrained(config.output_dir, state_dict=state_dict)

    # Save surgery config as yaml
    yaml.safe_dump(config.to_serialized(), (config.output_dir / "surgery_config.yaml").open("w"))
    logger.info("Done!")


def main(args=None):
    configure_logging()
    config: AddExpertsConfig = AddExpertsConfig.from_flat_args(args)
    add_experts(config)


if __name__ == "__main__":
    main()
