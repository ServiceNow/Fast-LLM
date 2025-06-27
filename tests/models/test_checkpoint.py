import functools
import pathlib
import shutil

import pytest
import safetensors.torch
import torch
import transformers
import yaml

from fast_llm.engine.checkpoint.config import (
    CheckpointFormat,
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    DistributedCheckpointFormat,
    FastLLMCheckpointFormat,
    ModelConfigType,
)
from fast_llm.engine.checkpoint.convert import ConvertConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, ShardName
from fast_llm.utils import Assert
from tests.utils.compare_tensor_logs import CompareConfig, compare_logged_tensor
from tests.utils.model_configs import ModelTestingConfig, ModelTestingGroup

_WEIGHT_SHARD_SAVE_NAME = f"{ShardName.weights}_shard"

_CHECKPOINT_AND_EVAL_ARGS = [
    "training.checkpoint.interval=1",
    "training.evaluators.validation.interval=2",
    "training.evaluators.validation.evaluator.iterations=1",
]


@pytest.mark.model_testing_group(ModelTestingGroup.checkpoint)
def test_checkpoint_and_eval(run_test_script_for_all_models, model_testing_config):
    # A baseline config (single-gpu, bf16, flash-attn).
    run_test_script_for_all_models(_CHECKPOINT_AND_EVAL_ARGS)


def _prepare_resume_fn(test_path: pathlib.Path, compare_path: pathlib.Path):
    shutil.copytree(compare_path, test_path)
    shutil.rmtree(test_path / "checkpoint" / "2")
    assert (test_path / "checkpoint" / "1" / "ok").is_file()
    # TODO: Eval
    shutil.rmtree(test_path / "runs")


def _compare_resume_fn(test_path: pathlib.Path, compare_path: pathlib.Path):
    for artifact in ["init", "train_1"]:
        path = f"runs/0/artifacts/0/tensor_logs_{artifact}.pt"
        if not (test_path / path).is_file():
            shutil.copy(compare_path / path, test_path / path)


@pytest.mark.depends_on(on=["test_checkpoint_and_eval[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.checkpoint)
def test_resume(run_test_script_for_all_models):
    # Resume from iteration=1 and compare outputs with the baseline run.
    run_test_script_for_all_models(
        _CHECKPOINT_AND_EVAL_ARGS,
        compare=f"test_checkpoint_and_eval",
        prepare_fn=_prepare_resume_fn,
        compare_fn=_compare_resume_fn,
    )


@pytest.mark.depends_on(on=["test_checkpoint_and_eval[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.checkpoint)
def test_resume_frozen(run_test_script_for_all_models):
    # Resume with frozen mlp. No comparison.
    run_test_script_for_all_models(
        _CHECKPOINT_AND_EVAL_ARGS + ["model.base_model.transformer.mlp_lr_scale=0."],
        compare="test_checkpoint_and_eval",
        prepare_fn=_prepare_resume_fn,
        do_compare=False,
    )


def do_get_convert_path(
    to: type[CheckpointFormat] | None = None, from_: type[CheckpointFormat] | None = None, *, base_path: pathlib.Path
) -> pathlib.Path:
    if to is None or from_ is None:
        return base_path / "test_checkpoint_and_eval" / "checkpoint" / "2"
    return base_path / "test_convert_model" / f"{to.name}_from_{from_.name}"


@pytest.fixture(scope="module")
def get_convert_path(run_test_script_base_path):
    return functools.partial(do_get_convert_path, base_path=run_test_script_base_path)


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


def _compare_safetensor_files(
    reference_path: pathlib.Path, *other_paths: pathlib.Path, expected_keys: set[str] | None = None
):
    reference = safetensors.torch.load_file(reference_path)
    if expected_keys is None:
        expected_keys = set(reference.keys())
    else:
        Assert.geq(set(reference.keys()), expected_keys)

    for other_path in other_paths:
        other = safetensors.torch.load_file(other_path)
        Assert.eq(other.keys(), expected_keys)
        for key in expected_keys:
            Assert.all_equal(reference[key], other[key])


@pytest.mark.depends_on(on=["test_conversion[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_converted_round_trip(model_testing_config, get_convert_path):
    # Test that the various possible conversion paths yield identical results.
    _compare_safetensor_files(
        get_convert_path() / "rank_0.safetensors",
        get_convert_path(DistributedCheckpointFormat, FastLLMCheckpointFormat) / "rank_0.safetensors",
        get_convert_path(DistributedCheckpointFormat, model_testing_config.checkpoint_format) / "rank_0.safetensors",
        expected_keys={_WEIGHT_SHARD_SAVE_NAME},
    )
    _compare_safetensor_files(
        get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat) / "model_0.safetensors",
        get_convert_path(FastLLMCheckpointFormat, model_testing_config.checkpoint_format) / "model_0.safetensors",
    )
    _compare_safetensor_files(
        get_convert_path(model_testing_config.checkpoint_format, DistributedCheckpointFormat) / "model_0.safetensors",
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
            )
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

    reference_shard = safetensors.torch.load_file(get_convert_path() / "rank_0.safetensors", device="cuda")[
        _WEIGHT_SHARD_SAVE_NAME
    ]

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
        get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat),
        reference_config,
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
    # Test that Fast-LLM's Hugging Face wrapper produces the same results as the converted Hugging Face model.
    # TODO: Review test. Move to test_generate?
    fast_llm_path = get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat)
    hf_path = get_convert_path(model_testing_config.checkpoint_format, DistributedCheckpointFormat)
    model_ref = model_testing_config.huggingface_model_for_causal_lm_class.from_pretrained(
        CheckpointLoadConfig(
            path=get_convert_path(),
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    test_input = torch.randint(
        0, model_ref.config.fast_llm_config.base_model.vocab_size, size=(4, 100), dtype=torch.int64, device="cuda"
    )
    output_ref = model_ref(test_input)
    model_from_fast_llm = model_testing_config.huggingface_model_for_causal_lm_class.from_pretrained(fast_llm_path)
    model_from_hf = model_testing_config.huggingface_model_for_causal_lm_class.from_pretrained(
        CheckpointLoadConfig(
            path=hf_path,
            format=model_testing_config.checkpoint_format,
            load_config=ModelConfigType.model,
        )
    )
    errors = []
    compare = CompareConfig()
    auto_model = (
        transformers.AutoModel
        if model_testing_config.name in ("diffusion_llama", "dream")
        else transformers.AutoModelForCausalLM
    )
    model_as_hf = auto_model.from_pretrained(
        hf_path, trust_remote_code=model_testing_config.checkpoint_format.trust_remote_code
    ).cuda()
    for name, model in zip(
        ("From state dict", "From Huggingface", "Native Huggingface"),
        (model_from_fast_llm, model_from_hf, model_as_hf),
    ):
        print(name)
        output = model(test_input)
        # TODO: Make a generic comparison util.
        compare_logged_tensor(
            {"samples": output_ref.logits, "shape": output_ref.logits.shape, "step": 0},
            {"samples": output.logits, "shape": output.logits.shape, "step": 0},
            errors,
            name,
            "logits",
            compare,
        )

    if errors:
        for error in errors:
            print(error)
        raise ValueError(f"Comparison failed ({len(errors)} errors)")


@pytest.fixture(scope="module")
def load_and_save_parallel_base_path(run_test_script_base_path):
    return run_test_script_base_path / "test_load_and_save_parallel"


@pytest.mark.depends_on(
    on=[
        "test_load_pretrained[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_save_and_load_in_parallel(run_distributed_script_for_all_models, load_and_save_parallel_base_path):
    # Save and load checkpoints to and from various distributed configurations.
    # Combined in a single test to mitigate process creation overhead.
    # TODO: Test beyond 2 gpu configs?
    import tests.models.distributed_test_checkpoint

    run_distributed_script_for_all_models(
        [tests.models.distributed_test_checkpoint.__file__],
        base_path=load_and_save_parallel_base_path,
        num_gpus=2,
    )


@pytest.fixture(scope="module")
def parallel_checkpoint_names(model_testing_config):
    names = []
    for format_ in (DistributedCheckpointFormat, FastLLMCheckpointFormat, model_testing_config.checkpoint_format):
        names.extend(
            [
                f"load_pretrained_{format_.name}_in_dp2",
                f"load_pretrained_{format_.name}_in_tp2",
                f"load_pretrained_{format_.name}_in_stp2",
                f"load_pretrained_{format_.name}_in_pp2",
            ]
        )

    names.extend(
        [
            "load_pretrained_dp2_in_stp2",
            "load_pretrained_stp2_in_dp2",
            "load_pretrained_tp2_in_pp2",
            "load_pretrained_pp2_in_tp2",
        ]
    )
    return names


@pytest.mark.depends_on(on=["test_save_and_load_in_parallel[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_load_parallel_checkpoint_in_single_gpu(
    load_and_save_parallel_base_path, get_convert_path, load_and_compare_checkpoints, parallel_checkpoint_names
):
    # Test single-gpu loading of multi-gpu distributed checkpoints.
    reference_shard = safetensors.torch.load_file(get_convert_path() / "rank_0.safetensors", device="cuda")[
        _WEIGHT_SHARD_SAVE_NAME
    ]

    for name in parallel_checkpoint_names:
        load_and_compare_checkpoints(
            DistributedCheckpointFormat,
            load_and_save_parallel_base_path / name / DistributedCheckpointFormat.name,
            None,
            reference_shard,
        )


@pytest.mark.depends_on(on=["test_save_and_load_in_parallel[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_parallel_checkpoint_consistency(model_testing_config, load_and_save_parallel_base_path, get_convert_path):
    # Check the consistency of the checkpoints saved in `test_save_and_load_in_parallel`
    checkpoint_formats = (DistributedCheckpointFormat, FastLLMCheckpointFormat, model_testing_config.checkpoint_format)
    # Compare Distributed checkpoints
    for config in ("dp2", "tp2", "stp2", "pp2"):
        for rank in range(2):
            _compare_safetensor_files(
                *[
                    load_and_save_parallel_base_path
                    / f"load_pretrained_{format_.name}_in_{config}"
                    / DistributedCheckpointFormat.name
                    / f"rank_{rank}.safetensors"
                    for format_ in checkpoint_formats
                ]
            )


@pytest.mark.depends_on(on=["test_save_and_load_in_parallel[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_multi_gpu_fast_llm_checkpoint(
    model_testing_config, load_and_save_parallel_base_path, get_convert_path, parallel_checkpoint_names
):
    # Fast-LLM checkpoints are independent of the distributed configuration that saved it.
    # TODO: Check pipeline-parallel checkpoints (two files).
    _compare_safetensor_files(
        get_convert_path(FastLLMCheckpointFormat, DistributedCheckpointFormat) / f"model_0.safetensors",
        *[
            load_and_save_parallel_base_path / name / FastLLMCheckpointFormat.name / f"model_0.safetensors"
            for name in parallel_checkpoint_names
            if "in_pp2" not in name
        ],
    )
