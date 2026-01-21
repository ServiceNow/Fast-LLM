#!/usr/bin/env python3
from pathlib import Path

# Import Fast-LLM components
import fast_llm.models.auto  # noqa: F401 - registers model types
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, DistributedCheckpointFormat, ModelConfigType
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration

model_path = "/mnt/checkpoints_fml/fast-llm-tutorial/debug_distributed_checkpoint_loading/checkpoint/1"
hf_path = "/mnt/checkpoints_fml/fast-llm-tutorial/debug_distributed_checkpoint_loading/export/apriel2/1"


# Configure checkpoint loading
load_config = CheckpointLoadConfig(
    path=Path(model_path),
    format=DistributedCheckpointFormat,
    model_weights=True,
    optimizer_state=True,
    load_config=ModelConfigType.fast_llm,
)
load_config.setup(MultiModalModel.config_class)

meta = MultiModalModel.config_class.load_metadata(load_config)
print(
    "Diffs vs metadata:",
    meta.config.compare(MultiModalModel.config_class.from_dict(meta.config), log_fn=lambda msg: print("  ", msg)),
)

model = MultiModalModel.from_pretrained(load_config, mode=StageMode.training, use_cpu=False)

for stage in model.stages_on_device.values():
    stage.restore_parameters()

print("FAST_LLM MODEL PARAMETER: model.base_model.decoder[0].mixer.mixers.kda.q_proj.weight")
print(model.base_model.decoder[0].mixer.mixers.kda.q_proj.weight)

hf_model = Apriel2ForConditionalGeneration.from_pretrained(
    hf_path, dtype="auto", trust_remote_code=True, device_map="cpu"
)
print("HUGGINGFACE MODEL PARAMETER: model.decoder.blocks.0.mixer.mixers.kda.q_proj.weight")
print(hf_model.state_dict()["model.decoder.blocks.0.mixer.mixers.kda.q_proj.weight"])
