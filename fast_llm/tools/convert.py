import argparse
import json
import logging
import math
import pathlib
import typing

import torch

from fast_llm.config import Config, Field, config_class, skip_valid_if_none
from fast_llm.distributed import get_float_dtype
from fast_llm.engine.multi_stage.config import CheckpointConfig, CheckpointType, PretrainedCheckpointConfig, StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.functional.config import TritonConfig
from fast_llm.logging import configure_logging
from fast_llm.models.auto import model_registry
from fast_llm.models.gpt.model import GPTModel
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


@config_class()
class ConversionConfig(Config):
    __argparse_map__ = {
        "dtype": {"type": str},
    }

    input_type: CheckpointType = Field()
    output_type: CheckpointType = Field()
    input_path: pathlib.Path = Field()
    output_path: pathlib.Path = Field()
    model_type: str | None = Field(default=None)
    use_cpu: bool = Field(default=False)
    exist_ok: bool = Field(default=False)
    target_params_per_file: int = Field(default=2**32)
    dtype: torch.dtype | None = Field(
        default=None,
        valid=skip_valid_if_none(get_float_dtype),
    )
    layers_per_step: int | None = Field(default=None)
    # TODO: Support other models (It's a classvar because Config doesn't support `type` variable).
    fast_llm_model_class: typing.ClassVar[type[FastLLMModel]] = GPTModel


def _convert_model_partial(
    model_class: type[FastLLMModel],
    config: ConversionConfig,
    output_path: pathlib.Path,
    stage_filter: set | None = None,
):
    logger.info(f"Loading {config.input_type} checkpoint from {config.input_path}...")
    model = config.fast_llm_model_class.from_pretrained(
        PretrainedCheckpointConfig(
            pretrained_checkpoint_path=config.input_path,
            pretrained_checkpoint_type=config.input_type,
            imported_model_type=config.model_type,
        ),
        mode=StageMode.weights,
        use_cpu=config.use_cpu,
        stage_filter=stage_filter,
    )
    logger.info(f"Saving {config.output_type} checkpoint to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=config.exist_ok)
    model.save_checkpoint(
        CheckpointConfig(
            checkpoint_path=output_path,
            checkpoint_type=config.output_type,
            exported_model_type=config.model_type,
            save_optimizer=False,
            target_params_per_file=config.target_params_per_file,
            dtype=config.dtype,
        )
    )
    (output_path / "ok").open("w")
    logger.info(f"Done!")


def convert_model(model_class: type[FastLLMModel], config: ConversionConfig):
    # TODO: Set logging in tests
    logging.getLogger().setLevel(logging.INFO)
    config.show()
    # Disable Triton to convert model on CPU
    if config.use_cpu:
        TritonConfig.TRITON_ENABLED = False
    # Skip on exist_ok=False if the model has already been processed
    if not config.exist_ok and (config.output_path / "ok").exists():
        logger.info(
            f"Output path {config.output_path} already exists and has been processed. Skipping model conversion..."
        )
        return
    if config.layers_per_step is None:
        _convert_model_partial(model_class, config, config.output_path)
    else:
        # TODO: Support other types?
        assert config.output_type == CheckpointType.huggingface
        logger.info(f">>> Loading model config")
        # Create a dummy version to determine the stage split.
        model = config.fast_llm_model_class.from_pretrained(
            PretrainedCheckpointConfig(
                pretrained_checkpoint_path=config.input_path,
                pretrained_checkpoint_type=config.input_type,
                imported_model_type=config.model_type,
                load_pretrained_weights=False,
            ),
            mode=StageMode.off_device,
            use_cpu=config.use_cpu,
        )
        stages_per_step = math.ceil(config.layers_per_step / model._multi_stage_config.layers_per_stage)
        num_stages = len(model.stages)
        step_paths = []
        for step_begin in range(0, num_stages, stages_per_step):
            step_end = min(step_begin + stages_per_step, num_stages)
            logger.info(f">>> Converting stages {step_begin} to {step_end-1} of {num_stages}")
            step_path = config.output_path / str(step_begin)
            step_paths.append(step_path)
            _convert_model_partial(model_class, config, step_path, set(range(step_begin, step_end)))
        logger.info(f">>> Aggregating conversion steps")

        # Combine weight maps and rename data files to avoid duplications.
        index_filename = "model.safetensors.index.json"
        config_filename = "config.json"
        index = {}
        weight_map = {}
        global_rename_map = {}
        file_count = 0
        for step_path in step_paths:
            step_index = json.load((step_path / index_filename).open("r"))
            if len(index) == 0:
                index.update(step_index)
                index["weight_map"] = weight_map
            step_weight_map = step_index["weight_map"]
            rename_map = {}
            for name, file_name in step_weight_map.items():
                if file_name in rename_map:
                    new_file_name = rename_map[file_name]
                else:
                    new_file_name = f"model_{file_count}.safetensors"
                    file_count += 1
                    rename_map[file_name] = new_file_name
                    global_rename_map[step_path / file_name] = config.output_path / new_file_name
                Assert.not_incl(name, weight_map)
                weight_map[name] = new_file_name

        # Save the combined index
        path = config.output_path / index_filename

        # Save the index.
        json.dump(index, path.open("w"), indent=4)

        # Copy the config
        (step_paths[0] / config_filename).rename(config.output_path / config_filename)

        # Move the data files
        for old_file_name, new_file_name in global_rename_map.items():
            old_file_name.rename(new_file_name)

        # Remove the remaining files
        for step_path in step_paths:
            (step_path / config_filename).unlink(missing_ok=True)
            (step_path / index_filename).unlink()
            (step_path / "ok").unlink()
            step_path.rmdir()

        # All good!
        (config.output_path / "ok").open("w")
        logger.info(f">>> All done!")


def convert(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type",
        choices=model_registry.keys(),
        help="The Fast-LLM model type to use. Must be defined in the model registry in `fast_llm.models.auto`.",
    )
    parsed, unparsed = parser.parse_known_args(args)
    model_class = model_registry[parsed.model_type]
    configure_logging()
    config: ConversionConfig = ConversionConfig.from_args(unparsed)
    convert_model(model_class, config)


if __name__ == "__main__":
    convert()
