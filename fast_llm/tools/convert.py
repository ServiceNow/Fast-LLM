import argparse
import functools
import json
import logging
import math
import typing
import warnings

from fast_llm.config import Field, config_class
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, CheckpointSaveConfig
from fast_llm.engine.checkpoint.external import HuggingfaceStateDictConverter
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageMode
from fast_llm.functional.config import TritonConfig
from fast_llm.models.auto import model_registry
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel

logger = logging.getLogger(__name__)


@config_class()
class ConversionConfig(RunnableConfig):
    input: CheckpointLoadConfig = Field(default_factory=CheckpointLoadConfig)
    output: CheckpointSaveConfig = Field(default_factory=CheckpointSaveConfig)
    use_cpu: bool = Field(default=False)
    exist_ok: bool = Field(default=False)
    layers_per_step: int | None = Field(default=None)

    @classmethod
    def _from_dict(
        cls,
        default: dict[str, typing.Any],
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove.
        if "input" not in default:
            default["input"] = {}
        if "output" not in default:
            default["output"] = {}
        if "input_type" in default:
            warnings.warn("`input_type` is deprecated. Use `input.format` instead.")
            default["input"]["format"] = default.pop("input_type")
        if "input_path" in default:
            warnings.warn("`input_path` is deprecated. Use `input.path` instead.")
            default["input"]["path"] = default.pop("input_path")
        if "output_type" in default:
            warnings.warn("`output_type` is deprecated. Use `output.format` instead.")
            default["output"]["format"] = default.pop("output_type")
        if "output_path" in default:
            warnings.warn("`output_path` is deprecated. Use `output.path` instead.")
            default["output"]["path"] = default.pop("output_path")
        if "model_type" in default:
            warnings.warn("`model_type` is deprecated. Use `output.format` instead.")
            # Will be handled in CheckpointConfigBase.from_dict
            default["input"]["model_type"] = default.pop("model_type")
            default["output"]["model_type"] = default.pop("model_type")
        if "target_params_per_file" in default:
            warnings.warn("`target_params_per_file` is deprecated. Use `output.parameters_per_file` instead.")
            default["output"]["parameters_per_file"] = default.pop("target_params_per_file")
        if "dtype" in default:
            warnings.warn("`dtype` is deprecated. Use `output.data_type` instead.")
            default["data_type"]["parameters_per_file"] = default.pop("dtype")
        return super()._from_dict(default, strict, flat)

    @classmethod
    def _get_parser(cls):
        parser = super()._get_parser()
        parser.add_argument(
            "model_type",
            choices=model_registry.keys(),
            help="The Fast-LLM model type to use. Must be defined in the model registry in `fast_llm.models.auto`.",
        )
        return parser

    def _get_runnable(self, parsed: argparse.Namespace) -> typing.Callable[[], None]:
        return functools.partial(self.run, parsed.model_type)

    def _convert_model_partial(
        self,
        model_class: type["FastLLMModel"],
        output: CheckpointSaveConfig,
        stage_filter: set | None = None,
    ):
        logger.info(f"Loading {self.input.format} checkpoint from {self.input.path}...")
        model = model_class.from_pretrained(
            self.input,
            mode=StageMode.weights,
            use_cpu=self.use_cpu,
            stage_filter=stage_filter,
        )
        logger.info(f"Saving {output.format} checkpoint to {output.path}...")
        output.path.mkdir(parents=True, exist_ok=self.exist_ok)
        model.save_checkpoint(output)
        (output.path / "ok").open("w")
        logger.info(f"Done!")

    def run(self, model_config_class: type["FastLLMModelConfig"] | str):
        # TODO: Set logging in tests
        logging.getLogger().setLevel(logging.INFO)
        self.to_logs()
        # Disable Triton to convert model on CPU
        if self.use_cpu:
            TritonConfig.TRITON_ENABLED = False
        # Skip on exist_ok=False if the model has already been processed
        if not self.exist_ok and (self.output.path / "ok").exists():
            logger.info(
                f"Output path {self.output.path} already exists and has been processed. Skipping model conversion..."
            )
            return
        if isinstance(model_config_class, str):
            model_config_class = model_registry[model_config_class]
        model_class = model_config_class.get_model_class()
        if self.layers_per_step is None:
            self._convert_model_partial(model_class, self.output)
        else:
            converter_class = model_config_class.get_converter_class(self.output.format)
            # TODO: Support other types?
            assert issubclass(converter_class, HuggingfaceStateDictConverter)
            logger.info(f">>> Loading model config")
            # Create a dummy version to determine the stage split.
            model = model_class.from_pretrained(
                self.input.to_copy({"model_weights": False}),
                mode=StageMode.off_device,
                use_cpu=self.use_cpu,
            )
            stages_per_step = math.ceil(self.layers_per_step / model._multi_stage_config.layers_per_stage)
            num_stages = len(model.stages)
            step_paths = []
            for step_begin in range(0, num_stages, stages_per_step):
                step_end = min(step_begin + stages_per_step, num_stages)
                logger.info(f">>> Converting stages {step_begin} to {step_end-1} of {num_stages}")
                step_path = self.output.path / str(step_begin)
                step_paths.append(step_path)
                self._convert_model_partial(
                    model_class, self.output.to_copy({"path": step_path}), set(range(step_begin, step_end))
                )
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
                        global_rename_map[step_path / file_name] = self.output.path / new_file_name
                    Assert.not_incl(name, weight_map)
                    weight_map[name] = new_file_name

            # Save the combined index
            path = self.output.path / index_filename

            # Save the index.
            json.dump(index, path.open("w"), indent=4)

            # Copy the config
            (step_paths[0] / config_filename).rename(self.output.path / config_filename)

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
            (self.output.path / "ok").open("w")
            logger.info(f">>> All done!")


if __name__ == "__main__":
    ConversionConfig.parse_and_run()
