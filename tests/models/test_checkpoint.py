import pathlib
import shutil

import pytest
import safetensors.torch
import torch
import transformers
import yaml

from fast_llm.engine.checkpoint.config import (
    CheckpointLoadConfig,
    CheckpointSaveConfig,
    DistributedCheckpointFormat,
    FastLLMCheckpointFormat,
    ModelConfigType,
)
from fast_llm.engine.checkpoint.convert import ConvertConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, ShardName, StageMode
from tests.utils.compare_tensor_logs import CompareConfig, compare_logged_tensor
from tests.utils.model_configs import ModelTestingGroup

_WEIGHT_SHARD_SAVE_NAME = f"{ShardName.weights}_shard"


@pytest.mark.model_testing_group(ModelTestingGroup.checkpoint)
def test_checkpoint_and_eval(run_test_script_for_all_models, model_testing_config):
    # A baseline config (single-gpu, bf16, flash-attn).
    run_test_script_for_all_models(
        model_testing_config.config_args
        + [
            "training.checkpoint.interval=1",
            "training.evaluators.validation.interval=2",
            "training.evaluators.validation.evaluator.iterations=1",
        ],
    )


def _prepare_resume_fn(test_path: pathlib.Path, compare_path: pathlib.Path, skip: bool) -> bool:
    if skip and (test_path / "checkpoint" / "2" / "ok").is_file():
        return True
    elif test_path.is_dir():
        shutil.rmtree(test_path)
    shutil.copytree(compare_path, test_path)
    shutil.rmtree(test_path / "checkpoint" / "2")
    assert (test_path / "checkpoint" / "1" / "ok").is_file()
    # TODO: Eval
    shutil.rmtree(test_path / "runs")
    return False


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
        [
            "training.checkpoint.interval=1",
            "training.evaluators.validation.interval=2",
            "training.evaluators.validation.evaluator.iterations=1",
        ],
        compare=f"test_checkpoint_and_eval",
        prepare_fn=_prepare_resume_fn,
        compare_fn=_compare_resume_fn,
    )


@pytest.mark.depends_on(on=["test_checkpoint_and_eval[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.checkpoint)
def test_resume_frozen(run_test_script_for_all_models):
    # Resume with frozen mlp. No comparison.
    run_test_script_for_all_models(
        [
            "training.checkpoint.interval=1",
            "training.evaluators.validation.interval=2",
            "training.evaluators.validation.evaluator.iterations=1",
            "model.base_model.transformer.mlp_lr_scale=0.",
        ],
        compare="test_checkpoint_and_eval",
        prepare_fn=_prepare_resume_fn,
        do_compare=False,
    )


def _run_conversion(config: ConvertConfig):
    if config.output.path.exists():
        assert config.output.path.is_dir()
        shutil.rmtree(config.output.path)
    config.run()


@pytest.fixture(scope="module")
def convert_paths(run_test_script_base_path):
    return {
        "checkpoint": run_test_script_base_path / "test_checkpoint_and_eval" / "checkpoint" / "2",
        "distributed_0": run_test_script_base_path / "test_convert_model" / "distributed_0",
        "distributed_1": run_test_script_base_path / "test_convert_model" / "distributed_1",
        "fast_llm_0": run_test_script_base_path / "test_convert_model" / "fast_llm_0",
        "fast_llm_1": run_test_script_base_path / "test_convert_model" / "fast_llm_1",
        "huggingface_0": run_test_script_base_path / "test_convert_model" / "huggingface_0",
        "huggingface_1": run_test_script_base_path / "test_convert_model" / "huggingface_1",
    }


@pytest.mark.depends_on(on=["test_checkpoint_and_eval[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_convert_distributed_to_fast_llm(model_testing_config, convert_paths):
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=convert_paths["checkpoint"],
                format=DistributedCheckpointFormat,
            ),
            output=CheckpointSaveConfig(
                path=convert_paths["fast_llm_0"],
                format=FastLLMCheckpointFormat,
            ),
            model=model_testing_config.model_config_class,
        )
    )


@pytest.mark.depends_on(on=["test_convert_distributed_to_fast_llm[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_convert_fast_llm_to_huggingface(model_testing_config, convert_paths):
    if model_testing_config.checkpoint_format is None:
        pytest.skip(f"Conversion not supported for {model_testing_config.name}")
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=convert_paths["fast_llm_0"],
                format=FastLLMCheckpointFormat,
            ),
            output=CheckpointSaveConfig(
                path=convert_paths["huggingface_0"],
                format=model_testing_config.checkpoint_format,
            ),
            model=model_testing_config.model_config_class,
        )
    )


@pytest.mark.depends_on(on=["test_convert_fast_llm_to_huggingface[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_convert_huggingface_to_distributed(model_testing_config, convert_paths):
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=convert_paths["huggingface_0"],
                format=model_testing_config.checkpoint_format,
            ),
            output=CheckpointSaveConfig(
                path=convert_paths["distributed_0"],
                format=DistributedCheckpointFormat,
            ),
            model=model_testing_config.model_config_class,
        )
    )


@pytest.mark.depends_on(on=["test_checkpoint_and_eval[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_convert_distributed_to_huggingface(model_testing_config, convert_paths):
    if model_testing_config.checkpoint_format is None:
        pytest.skip(f"Conversion not supported for {model_testing_config.name}")
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=convert_paths["checkpoint"],
                format=DistributedCheckpointFormat,
            ),
            output=CheckpointSaveConfig(
                path=convert_paths["huggingface_1"],
                format=model_testing_config.checkpoint_format,
            ),
            model=model_testing_config.model_config_class,
        )
    )


@pytest.mark.depends_on(on=["test_convert_distributed_to_huggingface[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_convert_huggingface_to_fast_llm(model_testing_config, convert_paths):
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=convert_paths["huggingface_1"],
                format=model_testing_config.checkpoint_format,
            ),
            output=CheckpointSaveConfig(
                path=convert_paths["fast_llm_1"],
                format=FastLLMCheckpointFormat,
            ),
            model=model_testing_config.model_config_class,
        )
    )


@pytest.mark.depends_on(on=["test_convert_huggingface_to_fast_llm[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_convert_fast_llm_to_distributed(model_testing_config, convert_paths):
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=convert_paths["fast_llm_1"],
                format=FastLLMCheckpointFormat,
            ),
            output=CheckpointSaveConfig(
                path=convert_paths["distributed_1"],
                format=DistributedCheckpointFormat,
            ),
            model=model_testing_config.model_config_class,
        )
    )


@pytest.mark.depends_on(
    on=[
        "test_convert_huggingface_to_distributed[{model_testing_config}]",
        "test_convert_fast_llm_to_distributed[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_converted_distributed(convert_paths):
    # Compare the fast llm weights
    # TODO: Compare configs
    w = safetensors.torch.load_file(convert_paths["checkpoint"] / "rank_0.safetensors")
    w0 = safetensors.torch.load_file(convert_paths["distributed_0"] / "rank_0.safetensors")
    w1 = safetensors.torch.load_file(convert_paths["distributed_1"] / "rank_0.safetensors")
    assert w.keys() >= {_WEIGHT_SHARD_SAVE_NAME}
    assert w0.keys() == w1.keys() == {_WEIGHT_SHARD_SAVE_NAME}
    for key in w0:
        assert w[key].shape == w0[key].shape, (key, w[key].shape, w0[key].shape)
        assert (w[key] == w0[key]).all(), (w[key], w0[key])
        assert w[key].shape == w1[key].shape, (key, w[key].shape, w1[key].shape)
        assert (w[key] == w1[key]).all(), (w[key], w1[key])


@pytest.mark.depends_on(
    on=[
        "test_convert_distributed_to_fast_llm[{model_testing_config}]",
        "test_convert_huggingface_to_fast_llm[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_converted_fast_llm(convert_paths):
    s0 = safetensors.torch.load_file(convert_paths["fast_llm_0"] / "model_0.safetensors")
    s1 = safetensors.torch.load_file(convert_paths["fast_llm_1"] / "model_0.safetensors")
    assert s0.keys() == s1.keys()
    for key in s0:
        assert s0[key].shape == s1[key].shape, (key, s0[key].shape, s1[key].shape)
        assert (s0[key] == s1[key]).all(), (key, s0, s1)


@pytest.mark.depends_on(
    on=[
        "test_convert_fast_llm_to_huggingface[{model_testing_config}]",
        "test_convert_distributed_to_huggingface[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_converted_huggingface(convert_paths):
    h0 = safetensors.torch.load_file(convert_paths["huggingface_0"] / "model_0.safetensors")
    h1 = safetensors.torch.load_file(convert_paths["huggingface_1"] / "model_0.safetensors")
    assert h0.keys() == h1.keys()
    for key in h0:
        assert h0[key].shape == h1[key].shape, (key, h0[key].shape, h1[key].shape)
        assert (h0[key] == h1[key]).all()


def _compare_model_configs(config_ref: FastLLMModelConfig, config_test: FastLLMModelConfig):
    config_ref.base_model.compare(config_test.base_model)


def _compare_architectures(config_ref: FastLLMModelConfig, config_test: FastLLMModelConfig):
    config_ref.base_model.compare_architecture(config_test.base_model)


@pytest.mark.depends_on(on=["test_converted_distributed[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_load_pretrained_distributed_checkpoint(model_testing_config, convert_paths):
    config = model_testing_config.model_config_class.from_dict(
        yaml.safe_load((convert_paths["checkpoint"] / ".." / ".." / "config.yaml").open("r"))["model"], strict=False
    )
    pretrained_config_ref = CheckpointLoadConfig(
        path=convert_paths["checkpoint"],
        format=DistributedCheckpointFormat,
        optimizer_state=True,
        load_config=ModelConfigType.model,
    )
    model = model_testing_config.model_class.from_pretrained(pretrained_config_ref)
    _compare_model_configs(config, model.config)
    state_shards = safetensors.torch.load_file(
        convert_paths["checkpoint"] / "rank_0.safetensors", device=str(model._distributed.device)
    )
    for shard_name in model.state_shard_names:
        assert (state_shards[f"{shard_name}_shard"] == model.get_shard(shard_name)).all()


@pytest.mark.depends_on(on=["test_load_pretrained_distributed_checkpoint[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_load_converted_distributed_checkpoint(model_testing_config, convert_paths):
    config_ref = model_testing_config.model_config_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["checkpoint"],
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )

    model = model_testing_config.model_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["distributed_0"],
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    config_alt = model_testing_config.model_config_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["distributed_1"],
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    _compare_architectures(config_ref, model.config)
    _compare_model_configs(model.config, config_alt)
    weight_shard = safetensors.torch.load_file(
        convert_paths["checkpoint"] / "rank_0.safetensors", device=str(model._distributed.device)
    )[_WEIGHT_SHARD_SAVE_NAME]
    assert (weight_shard == model.get_shard(ShardName.weights)).all()


@pytest.mark.depends_on(
    on=[
        "test_converted_fast_llm[{model_testing_config}]",
        "test_load_pretrained_distributed_checkpoint[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_load_converted_fast_llm_checkpoint(model_testing_config, convert_paths):
    config_ref = model_testing_config.model_config_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["checkpoint"],
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    model = model_testing_config.model_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["fast_llm_0"],
            format=FastLLMCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    config_alt = model_testing_config.model_config_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["fast_llm_1"],
            format=FastLLMCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    _compare_architectures(config_ref, model.config)
    _compare_architectures(config_ref, config_alt)
    weight_shard = safetensors.torch.load_file(
        convert_paths["checkpoint"] / "rank_0.safetensors", device=str(model._distributed.device)
    )[_WEIGHT_SHARD_SAVE_NAME]
    assert (weight_shard == model.get_shard(ShardName.weights)).all()


@pytest.mark.depends_on(
    on=[
        "test_converted_fast_llm[{model_testing_config}]",
        "test_load_pretrained_distributed_checkpoint[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_load_converted_huggingface_checkpoint(model_testing_config, convert_paths):
    config_ref = model_testing_config.model_config_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["checkpoint"],
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    model = model_testing_config.model_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["huggingface_1"],
            format=model_testing_config.checkpoint_format,
            load_config=ModelConfigType.model,
        ),
        mode=StageMode.weights,
    )
    config_alt = model_testing_config.model_config_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["huggingface_0"],
            format=model_testing_config.checkpoint_format,
            load_config=ModelConfigType.model,
        )
    )
    _compare_architectures(config_ref, model.config)
    _compare_model_configs(model.config, config_alt)
    weight_shard = safetensors.torch.load_file(
        convert_paths["checkpoint"] / "rank_0.safetensors", device=str(model._distributed.device)
    )[_WEIGHT_SHARD_SAVE_NAME]
    assert (weight_shard == model.get_shard(ShardName.weights)).all()


@pytest.mark.depends_on(
    on=[
        "test_load_converted_fast_llm_checkpoint[{model_testing_config}]",
        "test_load_converted_huggingface_checkpoint[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_run_converted_model(model_testing_config, convert_paths):
    model_ref = model_testing_config.huggingface_model_for_causal_lm_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["checkpoint"],
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    test_input = torch.randint(
        0, model_ref.config.fast_llm_config.base_model.vocab_size, size=(4, 100), dtype=torch.int64, device="cuda"
    )
    output_ref = model_ref(test_input)
    model_from_fast_llm = model_testing_config.huggingface_model_for_causal_lm_class.from_pretrained(
        convert_paths["fast_llm_0"]
    )
    model_from_hf = model_testing_config.huggingface_model_for_causal_lm_class.from_pretrained(
        CheckpointLoadConfig(
            path=convert_paths["huggingface_0"],
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
        convert_paths["huggingface_0"], trust_remote_code=model_testing_config.checkpoint_format.trust_remote_code
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


@pytest.mark.depends_on(on=["test_load_converted_distributed_checkpoint[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_load_pretrained_distributed_in_dp2(run_test_script_for_all_models, convert_paths):
    run_test_script_for_all_models(
        [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={convert_paths["distributed_0"]}",
            f"pretrained.format={DistributedCheckpointFormat.name}",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )


@pytest.mark.depends_on(on=["test_load_converted_distributed_checkpoint[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert)
def test_load_pretrained_distributed_with_config(run_test_script_for_all_models, convert_paths):
    run_test_script_for_all_models(
        [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={convert_paths["distributed_0"]}",
            f"pretrained.format={DistributedCheckpointFormat.name}",
            "schedule.skip_step=True",
        ],
    )


@pytest.mark.depends_on(on=["test_load_pretrained_distributed_in_dp2[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_load_pretrained_in_dp2_match_checkpoint(model_testing_config, convert_paths, run_test_script_base_path):
    test_ckpt_path = run_test_script_base_path / "test_load_pretrained_distributed_in_dp2" / "checkpoint" / "1"
    pretrained_config_ref = CheckpointLoadConfig(
        path=convert_paths["checkpoint"],
        format=DistributedCheckpointFormat,
        load_config=ModelConfigType.fast_llm,
    )
    pretrained_config_test = CheckpointLoadConfig(
        path=test_ckpt_path,
        format=DistributedCheckpointFormat,
        load_config=ModelConfigType.fast_llm,
    )
    config_ref = model_testing_config.model_config_class.from_pretrained(pretrained_config_ref)
    config_test = model_testing_config.model_config_class.from_pretrained(pretrained_config_test)
    _compare_model_configs(config_ref, config_test)
    shards_ref = safetensors.torch.load_file(convert_paths["checkpoint"] / "rank_0.safetensors")
    shards_test = [safetensors.torch.load_file(test_ckpt_path / f"rank_{i}.safetensors") for i in range(2)]
    ref_model = model_testing_config.model_class(config_ref)
    test_model = model_testing_config.model_class(config_test)

    weight_shard_ref_split = shards_ref[_WEIGHT_SHARD_SAVE_NAME].split(ref_model._stage_weight_shard_sizes)
    weight_shards_test_split = [
        shard_test[_WEIGHT_SHARD_SAVE_NAME].split(test_model._stage_weight_shard_sizes) for shard_test in shards_test
    ]
    for shard_test in shards_test:
        for shard_name, shard in shard_test.items():
            if shard_name != _WEIGHT_SHARD_SAVE_NAME:
                assert (shard == 0).all()  # noqa

    assert len(ref_model._stage_weight_shard_sizes) == len(test_model._stage_weight_shard_sizes)
    for i, stage_shard_ref in enumerate(weight_shard_ref_split):
        assert (
            test_model._stage_weight_shard_sizes[i]
            == ref_model._stage_weight_shard_sizes[i] // 2 + (-ref_model._stage_weight_shard_sizes[i] // 2) % 32
        )

        stage_shard_test = torch.concatenate(
            [weight_shard_test_split[i] for weight_shard_test_split in weight_shards_test_split]
        )
        assert (stage_shard_test[: stage_shard_ref.numel()] == stage_shard_ref).all()
        assert (stage_shard_test[stage_shard_ref.numel() :] == 0).all()  # noqa


@pytest.mark.depends_on(on=["test_load_pretrained_in_dp2_match_checkpoint[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_load_distributed_checkpoint_dp2(model_testing_config, convert_paths, run_test_script_base_path):
    # This also tests conversion which uses `FastLLMModel.from_checkpoint`
    pretrained_config_ref = CheckpointLoadConfig(
        path=convert_paths["checkpoint"],
        format=DistributedCheckpointFormat,
        load_config=ModelConfigType.fast_llm,
    )
    pretrained_config_test = CheckpointLoadConfig(
        path=run_test_script_base_path / "test_load_pretrained_distributed_in_dp2" / "checkpoint" / "1",
        format=DistributedCheckpointFormat,
        load_config=ModelConfigType.model,
    )
    config = model_testing_config.model_config_class.from_pretrained(pretrained_config_ref)
    model = model_testing_config.model_class.from_pretrained(pretrained_config_test, mode=StageMode.weights)
    _compare_model_configs(config, model.config)
    weight_shard = safetensors.torch.load_file(
        convert_paths["checkpoint"] / "rank_0.safetensors", device=str(model._distributed.device)
    )[_WEIGHT_SHARD_SAVE_NAME]
    assert (weight_shard == model.get_shard(ShardName.weights)).all()


@pytest.mark.depends_on(
    on=[
        "test_load_converted_fast_llm_checkpoint[{model_testing_config}]",
        "test_load_pretrained_in_dp2_match_checkpoint[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_load_pretrained_fast_llm_in_dp2(run_test_script_for_all_models, convert_paths, run_test_script_base_path):
    run_test_script_for_all_models(
        [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={convert_paths["fast_llm_0"]}",
            f"pretrained.format=fast_llm",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )
    for rank in range(2):
        ref_shard = safetensors.torch.load_file(
            run_test_script_base_path
            / f"test_load_pretrained_distributed_in_dp2"
            / "checkpoint"
            / "1"
            / f"rank_{rank}.safetensors"
        )
        test_shard = safetensors.torch.load_file(
            run_test_script_base_path
            / f"test_load_pretrained_fast_llm_in_dp2"
            / "checkpoint"
            / "1"
            / f"rank_{rank}.safetensors"
        )
        for name in set(ref_shard) | set(test_shard):
            assert (ref_shard[name] == test_shard[name]).all()


@pytest.mark.depends_on(
    on=[
        "test_load_converted_huggingface_checkpoint[{model_testing_config}]",
        "test_load_pretrained_in_dp2_match_checkpoint[{model_testing_config}]",
    ]
)
@pytest.mark.model_testing_group(ModelTestingGroup.convert, ModelTestingGroup.distributed)
def test_load_pretrained_huggingface_in_dp2(
    run_test_script_for_all_models, model_testing_config, run_test_script_base_path, convert_paths
):
    run_test_script_for_all_models(
        [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={convert_paths["huggingface_0"]}",
            f"pretrained.format={model_testing_config.checkpoint_format.name}",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )
    for rank in range(2):
        ref_shard = safetensors.torch.load_file(
            run_test_script_base_path
            / f"test_load_pretrained_distributed_in_dp2"
            / "checkpoint"
            / "1"
            / f"rank_{rank}.safetensors"
        )
        test_shard = safetensors.torch.load_file(
            run_test_script_base_path
            / f"test_load_pretrained_huggingface_in_dp2"
            / "checkpoint"
            / "1"
            / f"rank_{rank}.safetensors"
        )
        for name in set(ref_shard) | set(test_shard):
            assert (ref_shard[name] == test_shard[name]).all()
