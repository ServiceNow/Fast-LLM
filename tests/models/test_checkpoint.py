import gc
import logging
import pathlib
import shutil

import pytest
import safetensors.torch
import torch
import yaml

from fast_llm.config import NoAutoValidate
from fast_llm.engine.checkpoint.config import (
    CheckpointFormat,
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    DistributedCheckpointFormat,
    FastLLMCheckpointFormat,
    ModelConfigType,
)
from fast_llm.engine.checkpoint.convert import ConvertConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, ShardName, StageMode
from fast_llm.utils import Assert, header
from tests.utils.compare_tensor_logs import CompareConfig
from tests.utils.distributed_configs import DistributedTestingConfig
from tests.utils.model_configs import ModelTestingConfig, ModelTestingGroup
from tests.utils.save_load_configs import DISTRIBUTED_SAVE_LOAD_CONFIGS, DistributedSaveLoadConfig
from tests.utils.subtest import DistributedTestContext

logger = logging.getLogger(__name__)

_WEIGHT_SHARD_SAVE_NAME = f"{ShardName.weights}_shard"

_CHECKPOINT_AND_EVAL_ARGS = [
    "training.checkpoint.interval=1",
    "training.evaluators.validation.interval=2",
    "training.evaluators.validation.evaluator.iterations=1",
]


@pytest.mark.model_testing_group(ModelTestingGroup.checkpoint)
def test_checkpoint_and_eval(run_test_script_for_all_models, model_testing_config):
    # A baseline config (single-gpu, bf16, flash-attn).
    run_test_script_for_all_models(
        distributed_testing_config=DistributedTestingConfig(
            name="checkpoint_and_eval", config_args=_CHECKPOINT_AND_EVAL_ARGS
        ),
    )


@pytest.fixture(scope="module")
def prepare_resume(run_test_script_base_path: pathlib.Path):
    def do_prepare_resume(distributed_testing_config: DistributedTestingConfig):
        self_path = run_test_script_base_path / distributed_testing_config.name
        shutil.copytree(run_test_script_base_path / distributed_testing_config.compare, self_path)
        shutil.rmtree(self_path / "checkpoint" / "2")
        assert (self_path / "checkpoint" / "1" / "ok").is_file()
        # TODO: Eval
        shutil.rmtree(self_path / "runs")

    return do_prepare_resume


@pytest.mark.depends_on(on=["test_checkpoint_and_eval[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.checkpoint)
def test_resume(run_test_script_for_all_models, compare_results_for_all_models, prepare_resume):
    distributed_testing_config = DistributedTestingConfig(
        name="resume",
        compare="checkpoint_and_eval",
        config_args=_CHECKPOINT_AND_EVAL_ARGS,
        compare_config=CompareConfig(sub_configs={(("init", "train_1"), None): CompareConfig(ignore_tensors=True)}),
    )
    prepare_resume(distributed_testing_config)
    # Resume from iteration=1 and compare outputs with the baseline run.
    run_test_script_for_all_models(distributed_testing_config)
    compare_results_for_all_models(distributed_testing_config)


@pytest.mark.depends_on(on=["test_checkpoint_and_eval[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.checkpoint)
def test_resume_frozen(run_test_script_for_all_models, prepare_resume):
    # TODO: No more frozen weights?
    distributed_testing_config = DistributedTestingConfig(
        name="resume_frozen", compare="checkpoint_and_eval", config_args=_CHECKPOINT_AND_EVAL_ARGS
    )
    prepare_resume(distributed_testing_config)
    # Resume with frozen mlp. No comparison.
    run_test_script_for_all_models(distributed_testing_config)


@pytest.fixture(scope="module")
def run_conversion(model_testing_config: ModelTestingConfig, get_convert_path):
    def do_run_conversion(
        load_path: pathlib.Path, load_format: type[CheckpointFormat] | None, save_format: type[CheckpointFormat] | None
    ):
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=load_path,
                format=load_format,
            ),
            output=CheckpointSaveConfig(
                path=get_convert_path(save_format, load_format),
                format=save_format,
            ),
            use_cpu=not torch.cuda.is_available(),
            model=model_testing_config.model_config_class,
        ).run()

    return do_run_conversion


@pytest.mark.depends_on(on=["test_checkpoint_and_eval[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_conversion(model_testing_config, run_conversion, get_convert_path):
    # Test that the various conversions between formats complete successfully.
    run_conversion(
        get_convert_path(),
        DistributedCheckpointFormat,
        FastLLMCheckpointFormat,
    )
    if model_testing_config.checkpoint_format is None:
        run_conversion(
            get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat),
            FastLLMCheckpointFormat,
            DistributedCheckpointFormat,
        )
    else:
        run_conversion(
            get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat),
            FastLLMCheckpointFormat,
            model_testing_config.checkpoint_format,
        )
        run_conversion(
            get_convert_path(model_testing_config.checkpoint_format, FastLLMCheckpointFormat),
            model_testing_config.checkpoint_format,
            DistributedCheckpointFormat,
        )
        run_conversion(
            get_convert_path(),
            DistributedCheckpointFormat,
            model_testing_config.checkpoint_format,
        )
        run_conversion(
            get_convert_path(model_testing_config.checkpoint_format, DistributedCheckpointFormat),
            model_testing_config.checkpoint_format,
            FastLLMCheckpointFormat,
        )
        run_conversion(
            get_convert_path(FastLLMCheckpointFormat, model_testing_config.checkpoint_format),
            FastLLMCheckpointFormat,
            DistributedCheckpointFormat,
        )


def compare_safetensor_files(
    reference: pathlib.Path | dict[str, torch.Tensor],
    *other_paths: pathlib.Path,
    expected_keys: set[str] | None = None,
):
    if isinstance(reference, pathlib.Path):
        reference = safetensors.torch.load_file(reference)
    if expected_keys is None:
        expected_keys = set(reference.keys())
    else:
        Assert.geq(set(reference.keys()), expected_keys)

    for other_path in other_paths:
        other = safetensors.torch.load_file(other_path)
        if other.keys() != expected_keys:
            raise ValueError(f"Expected keys {expected_keys} but got {other.keys()} in {other_path}")
        for key in expected_keys:
            Assert.all_equal(reference[key], other[key], msg=f"tensor = {key}, path = {other_path}")


@pytest.mark.depends_on(on=["test_conversion[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_converted_round_trip(model_testing_config, get_convert_path):
    # Test that the various possible conversion paths yield identical results.
    if model_testing_config.checkpoint_format is None:
        compare_safetensor_files(
            get_convert_path() / "rank_0.safetensors",
            get_convert_path(DistributedCheckpointFormat, FastLLMCheckpointFormat) / "rank_0.safetensors",
            expected_keys={_WEIGHT_SHARD_SAVE_NAME},
        )
    else:
        compare_safetensor_files(
            get_convert_path() / "rank_0.safetensors",
            get_convert_path(DistributedCheckpointFormat, FastLLMCheckpointFormat) / "rank_0.safetensors",
            get_convert_path(DistributedCheckpointFormat, model_testing_config.checkpoint_format)
            / "rank_0.safetensors",
            expected_keys={_WEIGHT_SHARD_SAVE_NAME},
        )
        compare_safetensor_files(
            get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat) / "model_0.safetensors",
            get_convert_path(FastLLMCheckpointFormat, model_testing_config.checkpoint_format) / "model_0.safetensors",
        )
        compare_safetensor_files(
            get_convert_path(model_testing_config.checkpoint_format, DistributedCheckpointFormat)
            / "model_0.safetensors",
            get_convert_path(model_testing_config.checkpoint_format, FastLLMCheckpointFormat) / "model_0.safetensors",
        )


def _compare_model_configs(config_ref: FastLLMModelConfig, config_test: FastLLMModelConfig):
    config_ref.base_model.compare(config_test.base_model)


def _compare_architectures(config_ref: FastLLMModelConfig, config_test: FastLLMModelConfig):
    config_ref.base_model.compare_architecture(config_test.base_model)


@pytest.fixture(scope="module")
def load_and_compare_checkpoints(model_testing_config):
    def do_load_and_compare_checkpoints(
        load_format: type[CheckpointFormat], load_path: pathlib.Path, reference_config, reference_shard
    ):
        model = model_testing_config.model_class.from_pretrained(
            CheckpointLoadConfig(
                path=load_path,
                format=load_format,
            ),
            {("distributed", "use_cuda"): torch.cuda.is_available()},
        )
        if reference_config is not None:
            _compare_model_configs(reference_config, model.config)
        if reference_shard is not None:
            Assert.all_equal(model.get_shard(ShardName.weights), reference_shard)

    return do_load_and_compare_checkpoints


@pytest.mark.depends_on(on=["test_conversion[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_load_pretrained(
    model_testing_config, run_test_script_base_path, get_convert_path, load_and_compare_checkpoints
):
    # Test that loadind a pretrained model from either converted checkpoint always yields the exact same model.
    reference_config = model_testing_config.model_config_class.from_dict(
        yaml.safe_load(get_convert_path().parents[1].joinpath("config.yaml").open("r"))["model"]
    )
    reference_shard = safetensors.torch.load_file(
        get_convert_path() / "rank_0.safetensors", device="cuda" if torch.cuda.is_available() else "cpu"
    )[_WEIGHT_SHARD_SAVE_NAME]
    load_and_compare_checkpoints(
        FastLLMCheckpointFormat,
        get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat),
        reference_config,
        reference_shard,
    )
    if model_testing_config.checkpoint_format is None:
        load_and_compare_checkpoints(
            DistributedCheckpointFormat,
            get_convert_path(DistributedCheckpointFormat, FastLLMCheckpointFormat),
            reference_config,
            reference_shard,
        )
        return

    reference_config_from_hf = model_testing_config.model_config_class.from_dict(
        {
            "base_model": yaml.safe_load(
                get_convert_path(FastLLMCheckpointFormat, model_testing_config.checkpoint_format)
                .joinpath("metadata.yaml")
                .open("r")
            )["config"]["base_model"]
        }
    )
    _compare_architectures(reference_config, reference_config_from_hf)

    load_and_compare_checkpoints(DistributedCheckpointFormat, get_convert_path(), reference_config, reference_shard)

    load_and_compare_checkpoints(
        DistributedCheckpointFormat,
        get_convert_path(DistributedCheckpointFormat, FastLLMCheckpointFormat),
        reference_config_from_hf,
        reference_shard,
    )
    load_and_compare_checkpoints(
        DistributedCheckpointFormat,
        get_convert_path(DistributedCheckpointFormat, model_testing_config.checkpoint_format),
        reference_config_from_hf,
        reference_shard,
    )

    load_and_compare_checkpoints(
        FastLLMCheckpointFormat,
        get_convert_path(FastLLMCheckpointFormat, model_testing_config.checkpoint_format),
        reference_config_from_hf,
        reference_shard,
    )

    load_and_compare_checkpoints(
        model_testing_config.checkpoint_format,
        get_convert_path(model_testing_config.checkpoint_format, DistributedCheckpointFormat),
        reference_config_from_hf,
        reference_shard,
    )
    load_and_compare_checkpoints(
        model_testing_config.checkpoint_format,
        get_convert_path(model_testing_config.checkpoint_format, FastLLMCheckpointFormat),
        reference_config_from_hf,
        reference_shard,
    )


@pytest.mark.depends_on(on=["test_load_pretrained[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_huggingface_model(model_testing_config, get_convert_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distributed_update = {("distributed", "use_cuda"): torch.cuda.is_available()}
    if model_testing_config.checkpoint_format is None:
        return
    # Test that Fast-LLM's Hugging Face wrapper produces the same results as the converted Hugging Face model.
    # TODO: Stress the importance of this test as the main correctness test for most models.
    # TODO: Review test. Move to test_generate?
    fast_llm_path = get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat)
    hf_path = get_convert_path(model_testing_config.checkpoint_format, DistributedCheckpointFormat)
    try:
        hf_class = model_testing_config.huggingface_model_for_causal_lm_class
    except NotImplementedError:
        pytest.skip(f"Hugging Face wrapper not implemented for {model_testing_config.name}.")
    model_ref = hf_class.from_pretrained(
        CheckpointLoadConfig(
            path=get_convert_path(),
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        ),
        distributed_update,
    ).eval()
    test_input = torch.randint(
        0,
        384,
        size=(4, 100),
        dtype=torch.int64,
        device=device,
    )
    kwargs = {}
    if model_testing_config.model_type == "multimodal":
        kwargs["pixel_values"] = torch.rand([6, 3, 20, 20]).to(device)
        kwargs["image_sizes"] = torch.tensor(
            [
                [20, 20],  # Full image, 25 patches
                [12, 12],  # Smaller, 9 patches
                [9, 15],  # Cropped to patch size, 6 patches
                [5, 20],  # Cropped in one dim, 5 patches
                [7, 5],  # Single patch
                [2, 3],  # Cropped out (0 patch)
            ]
        )
        image_token_index = model_ref.fast_llm_base_model.config.image_token_index
        # First sample has one image at the beginning.
        test_input[0, :25] = image_token_index
        # Second sample has one image in the middle
        test_input[1, 30:39] = image_token_index
        # Third sample has no image.
        # Fourth sample has four images.
        # First one has discontinuous embedding (ex. image break token)
        test_input[3, :3] = image_token_index
        test_input[3, 7:10] = image_token_index
        # Second and third one next to each other.
        test_input[3, 28:34] = image_token_index
        # Last one cropped out.

    output_ref = model_ref(test_input, **kwargs)
    model_from_fast_llm = hf_class.from_pretrained(fast_llm_path, distributed_update).eval()
    model_from_hf = hf_class.from_pretrained(
        CheckpointLoadConfig(
            path=hf_path,
            format=model_testing_config.checkpoint_format,
            load_config=ModelConfigType.model,
        ),
        distributed_update,
    ).eval()
    errors = []
    model_as_hf = (
        model_testing_config.auto_model_class.from_pretrained(hf_path, trust_remote_code=True)
        .to("cuda" if torch.cuda.is_available() else "cpu")
        .eval()
    )
    for name, model in zip(
        ("From state dict", "From Huggingface", "Native Huggingface"),
        (model_from_fast_llm, model_from_hf, model_as_hf),
    ):
        print(name)
        output = model(test_input, **kwargs)
        # TODO: Make a generic comparison util.
        CompareConfig().compare_tensors(
            {"samples": output_ref.logits, "shape": output_ref.logits.shape, "step": 0},
            {"samples": output.logits, "shape": output.logits.shape, "step": 0},
            errors,
            name,
            "logits",
        )

    if errors:
        for error in errors:
            print(error)
        raise ValueError(f"Comparison failed ({len(errors)} errors)")


def _save_and_load_in_parallel(
    test_context: DistributedTestContext, base_path: pathlib.Path, model_testing_config: ModelTestingConfig
) -> None:
    # Import all dynamic classes.
    import fast_llm.cli  # noqa

    for config in DISTRIBUTED_SAVE_LOAD_CONFIGS.values():
        if config.load_format == "{checkpoint_format}" and model_testing_config.checkpoint_format is None:
            continue
        config = config.resolve(base_path, model_testing_config)
        with test_context.subtest(base_path, config.name, config.num_gpus) as subtest:
            if subtest.do_run:
                logger.info(header(config.name))
                logger.info(f"Loading {config.load_format} checkpoint from {config.load_path}")
                with NoAutoValidate():
                    load_config = CheckpointLoadConfig(path=config.load_path, format=config.load_format)
                load_config.setup(model_testing_config.model_config_class)
                load_config.validate()
                model = model_testing_config.model_class.from_pretrained(
                    load_config,
                    # The world size and rank are already set through environment variable.
                    {"distributed": config.distributed},
                    mode=StageMode.inference,
                )
                for save_format in (DistributedCheckpointFormat, FastLLMCheckpointFormat):
                    logger.info(f"Saving {save_format.name} checkpoint to {config.save_path / save_format.name}")
                    model.save_checkpoint(
                        CheckpointSaveConfig(path=config.save_path / save_format.name, format=save_format)
                    )
                del model
                gc.collect()
                torch.cuda.empty_cache()


@pytest.mark.depends_on(on=["test_load_pretrained[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_save_and_load_in_parallel(run_parallel_script, run_test_script_base_path, model_testing_config):
    # Save and load checkpoints to and from various distributed configurations.
    # Combined in a single test to mitigate process creation overhead.
    # TODO: Test beyond 2 gpu configs?
    if torch.cuda.device_count() < 2:
        pytest.skip(f"Not enough GPUs2")
    run_parallel_script(
        _save_and_load_in_parallel,
        (run_test_script_base_path, model_testing_config),
        world_size=2,
        backend=model_testing_config.distributed_backend,
    )


@pytest.fixture(scope="module")
def reference_distributed_shard(get_convert_path) -> torch.Tensor | None:
    # Load the file in a fixture (on cpu) so it's not loaded from disk each time.
    try:
        return safetensors.torch.load_file(get_convert_path() / "rank_0.safetensors")[_WEIGHT_SHARD_SAVE_NAME]
    except OSError:
        # The fixture may be evaluated even if the tests are to be skipped.
        return None


# We don't want to depend on `test_save_and_load_in_parallel` because we still want to run this in cas of failure.
# This should still run after `test_save_and_load_in_parallel`
@pytest.mark.depends_on(on=["test_load_pretrained[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_load_parallel_checkpoint_in_single_gpu(
    distributed_save_load_config: DistributedSaveLoadConfig,
    run_test_script_base_path,
    model_testing_config,
    load_and_compare_checkpoints,
    reference_distributed_shard,
    report_subtest,
):
    if (
        model_testing_config.checkpoint_format is None
        and distributed_save_load_config.load_format == "{checkpoint_format}"
    ):
        return
    # This should only happen when test is skipped (failed dependency).
    assert reference_distributed_shard is not None
    distributed_save_load_config = distributed_save_load_config.resolve(
        base_path=run_test_script_base_path, model_testing_config=model_testing_config
    )
    report_subtest(distributed_save_load_config.save_path, distributed_save_load_config.num_gpus)
    load_and_compare_checkpoints(
        DistributedCheckpointFormat,
        distributed_save_load_config.save_path / DistributedCheckpointFormat.name,
        None,
        reference_distributed_shard.to(device="cuda"),
    )


@pytest.mark.depends_on(on=["test_save_and_load_in_parallel[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_parallel_checkpoint_consistency(model_testing_config, run_test_script_base_path):
    # Check the consistency of the checkpoints saved in `test_save_and_load_in_parallel`
    # Compare Distributed checkpoints
    for config in ("dp2", "tp2", "stp2", "pp2"):
        for rank in range(2):
            compare_safetensor_files(
                *[
                    DISTRIBUTED_SAVE_LOAD_CONFIGS[f"load_{format_}_in_{config}"]
                    .resolve(base_path=run_test_script_base_path, model_testing_config=model_testing_config)
                    .save_path
                    / f"{DistributedCheckpointFormat.name}/rank_{rank}.safetensors"
                    for format_ in (DistributedCheckpointFormat.name, FastLLMCheckpointFormat.name)
                    + (() if model_testing_config.checkpoint_format is None else ("{checkpoint_format}",))
                ]
            )


@pytest.fixture(scope="module")
def reference_fast_llm_shard(get_convert_path) -> dict[str, torch.Tensor] | None:
    # Load the file in a fixture (on cpu) so it's not loaded from disk each time.
    try:
        return safetensors.torch.load_file(
            get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat) / f"model_0.safetensors"
        )
    except OSError:
        # The fixture may be evaluated even if the tests are to be skipped.
        return None


@pytest.mark.depends_on(on=["test_save_and_load_in_parallel[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_multi_gpu_fast_llm_checkpoint(
    model_testing_config, distributed_save_load_config_non_pp, run_test_script_base_path, reference_fast_llm_shard
):
    # This should only happen when test is skipped (failed dependency).
    assert reference_fast_llm_shard is not None
    # Fast-LLM checkpoints are independent of the distributed configuration that saved it.
    # TODO: Check pipeline-parallel checkpoints (two files).
    distributed_save_load_config_non_pp = distributed_save_load_config_non_pp.resolve(
        base_path=run_test_script_base_path, model_testing_config=model_testing_config
    )

    compare_safetensor_files(
        reference_fast_llm_shard,
        distributed_save_load_config_non_pp.save_path / f"{FastLLMCheckpointFormat.name}/model_0.safetensors",
    )
