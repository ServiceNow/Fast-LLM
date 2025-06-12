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
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, ShardName, StageMode
from fast_llm.models.auto import model_registry
from fast_llm.tools.convert import ConvertConfig
from tests.common import (
    CONFIG_COMMON,
    FORCE_REUSE_RESULTS,
    HUGGINGFACE_CHECKPOINT_FORMAT,
    REUSE_RESULTS,
    TEST_MODEL,
    TEST_MODEL_TYPE,
    TEST_RESULTS_PATH,
    requires_cuda,
)
from tests.compare_tensor_logs import CompareConfig, compare_logged_tensor

TEST_MODEL_CONFIG_CLS = model_registry[TEST_MODEL_TYPE]
TEST_MODEL_HF_CLS = TEST_MODEL_CONFIG_CLS.get_huggingface_model_for_causal_lm_class()
TEST_MODEL_CLS = TEST_MODEL_CONFIG_CLS.get_model_class()
TEST_BASE_MODEL_CONFIG_CLS = TEST_MODEL_CONFIG_CLS.get_base_model_config_class()

WEIGHT_SHARD_SAVE_NAME = f"{ShardName.weights}_shard"


@requires_cuda
def test_checkpoint_and_eval(run_test_script):
    # A baseline config (single-gpu, bf16, flash-attn).
    run_test_script(
        f"test_{TEST_MODEL}_checkpoint_and_eval",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.evaluations.validation.interval=2",
            "training.evaluations.validation.iterations=1",
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


@pytest.mark.depends(on=["test_checkpoint_and_eval"])
def test_resume(run_test_script):
    # Resume from iteration=1 and compare outputs with the baseline run.
    run_test_script(
        f"test_{TEST_MODEL}_resume",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.evaluations.validation.interval=2",
            "training.evaluations.validation.iterations=1",
        ],
        compare=f"test_{TEST_MODEL}_checkpoint_and_eval",
        prepare_fn=_prepare_resume_fn,
        compare_fn=_compare_resume_fn,
    )


@pytest.mark.depends(on=["test_checkpoint_and_eval"])
def test_resume_frozen(run_test_script):
    # Resume with frozen mlp. No comparison.
    run_test_script(
        f"test_{TEST_MODEL}_resume_frozen",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.evaluations.validation.interval=2",
            "training.evaluations.validation.iterations=1",
            "model.base_model.transformer.mlp_lr_scale=0.",
        ],
        compare=f"test_{TEST_MODEL}_checkpoint_and_eval",
        prepare_fn=_prepare_resume_fn,
        do_compare=False,
    )


def _run_conversion(config: ConvertConfig):
    if config.output.path.is_dir() and not REUSE_RESULTS:
        shutil.rmtree(config.output.path)
    if not config.output.path.is_dir():
        if FORCE_REUSE_RESULTS:
            raise RuntimeError(config.output.path)
        config.run()


_CKPT_PATH = TEST_RESULTS_PATH / f"test_{TEST_MODEL}_checkpoint_and_eval" / "checkpoint" / "2"
CONVERT_PATH = TEST_RESULTS_PATH / f"test_{TEST_MODEL}_convert_model"


@pytest.mark.depends(on=["test_checkpoint_and_eval"])
def test_convert_distributed_to_fast_llm():
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=_CKPT_PATH,
                format=DistributedCheckpointFormat,
            ),
            output=CheckpointSaveConfig(
                path=CONVERT_PATH / "fast_llm_0",
                format=FastLLMCheckpointFormat,
            ),
            model=TEST_MODEL_CONFIG_CLS,
        )
    )


@pytest.mark.depends(on=["test_convert_distributed_to_fast_llm"])
def test_convert_fast_llm_to_huggingface():
    if HUGGINGFACE_CHECKPOINT_FORMAT is None:
        pytest.skip(f"Conversion not supported for {TEST_MODEL}")
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=CONVERT_PATH / "fast_llm_0",
                format=FastLLMCheckpointFormat,
            ),
            output=CheckpointSaveConfig(
                path=CONVERT_PATH / "huggingface_0",
                format=HUGGINGFACE_CHECKPOINT_FORMAT,
            ),
            model=TEST_MODEL_CONFIG_CLS,
        )
    )


@pytest.mark.depends(on=["test_convert_fast_llm_to_huggingface"])
def test_convert_huggingface_to_distributed():
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=CONVERT_PATH / "huggingface_0",
                format=HUGGINGFACE_CHECKPOINT_FORMAT,
            ),
            output=CheckpointSaveConfig(
                path=CONVERT_PATH / "distributed_0",
                format=DistributedCheckpointFormat,
            ),
            model=TEST_MODEL_CONFIG_CLS,
        )
    )


@pytest.mark.depends(on=["test_checkpoint_and_eval"])
def test_convert_distributed_to_huggingface():
    if HUGGINGFACE_CHECKPOINT_FORMAT is None:
        pytest.skip(f"Conversion not supported for {TEST_MODEL}")
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=_CKPT_PATH,
                format=DistributedCheckpointFormat,
            ),
            output=CheckpointSaveConfig(
                path=CONVERT_PATH / "huggingface_1",
                format=HUGGINGFACE_CHECKPOINT_FORMAT,
            ),
            model=TEST_MODEL_CONFIG_CLS,
        )
    )


@pytest.mark.depends(on=["test_convert_distributed_to_huggingface"])
def test_convert_huggingface_to_fast_llm():
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=CONVERT_PATH / "huggingface_1",
                format=HUGGINGFACE_CHECKPOINT_FORMAT,
            ),
            output=CheckpointSaveConfig(
                path=CONVERT_PATH / "fast_llm_1",
                format=FastLLMCheckpointFormat,
            ),
            model=TEST_MODEL_CONFIG_CLS,
        )
    )


@pytest.mark.depends(on=["test_convert_huggingface_to_fast_llm"])
def test_convert_fast_llm_to_distributed():
    _run_conversion(
        ConvertConfig(
            input=CheckpointLoadConfig(
                path=CONVERT_PATH / "fast_llm_1",
                format=FastLLMCheckpointFormat,
            ),
            output=CheckpointSaveConfig(
                path=CONVERT_PATH / "distributed_1",
                format=DistributedCheckpointFormat,
            ),
            model=TEST_MODEL_CONFIG_CLS,
        )
    )


@pytest.mark.depends(on=["test_convert_huggingface_to_distributed", "test_convert_fast_llm_to_distributed"])
def test_converted_distributed():
    # Compare the fast llm weights
    # TODO: Compare configs
    w = safetensors.torch.load_file(_CKPT_PATH / "rank_0.safetensors")
    w0 = safetensors.torch.load_file(CONVERT_PATH / "distributed_0" / "rank_0.safetensors")
    w1 = safetensors.torch.load_file(CONVERT_PATH / "distributed_1" / "rank_0.safetensors")
    assert w.keys() >= {WEIGHT_SHARD_SAVE_NAME}
    assert w0.keys() == w1.keys() == {WEIGHT_SHARD_SAVE_NAME}
    for key in w0:
        assert w[key].shape == w0[key].shape, (key, w[key].shape, w0[key].shape)
        assert (w[key] == w0[key]).all(), (w[key], w0[key])
        assert w[key].shape == w1[key].shape, (key, w[key].shape, w1[key].shape)
        assert (w[key] == w1[key]).all(), (w[key], w1[key])


@pytest.mark.depends(on=["test_convert_distributed_to_fast_llm", "test_convert_huggingface_to_fast_llm"])
def test_converted_fast_llm():
    s0 = safetensors.torch.load_file(CONVERT_PATH / "fast_llm_0" / "model_0.safetensors")
    s1 = safetensors.torch.load_file(CONVERT_PATH / "fast_llm_1" / "model_0.safetensors")
    assert s0.keys() == s1.keys()
    for key in s0:
        assert s0[key].shape == s1[key].shape, (key, s0[key].shape, s1[key].shape)
        assert (s0[key] == s1[key]).all(), (key, s0, s1)


@pytest.mark.depends(on=["test_convert_fast_llm_to_huggingface", "test_convert_distributed_to_huggingface"])
def test_converted_huggingface():
    h0 = safetensors.torch.load_file(CONVERT_PATH / "huggingface_0" / "model_0.safetensors")
    h1 = safetensors.torch.load_file(CONVERT_PATH / "huggingface_1" / "model_0.safetensors")
    assert h0.keys() == h1.keys()
    for key in h0:
        assert h0[key].shape == h1[key].shape, (key, h0[key].shape, h1[key].shape)
        assert (h0[key] == h1[key]).all()


def _compare_model_configs(config_ref: FastLLMModelConfig, config_test: FastLLMModelConfig):
    config_ref.base_model.compare(config_test.base_model)


def _compare_architectures(config_ref: FastLLMModelConfig, config_test: FastLLMModelConfig):
    config_ref.base_model.compare_architecture(config_test.base_model)


@pytest.mark.depends(on=["test_converted_distributed"])
def test_load_pretrained_distributed_checkpoint():
    config = TEST_MODEL_CONFIG_CLS.from_dict(
        yaml.safe_load((_CKPT_PATH / ".." / ".." / "config.yaml").open("r"))["model"], strict=False
    )
    pretrained_config_ref = CheckpointLoadConfig(
        path=_CKPT_PATH,
        format=DistributedCheckpointFormat,
        optimizer_state=True,
        load_config=ModelConfigType.model,
    )
    model = TEST_MODEL_CLS.from_pretrained(pretrained_config_ref)
    _compare_model_configs(config, model.config)
    state_shards = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._distributed.device)
    )
    for shard_name in model.state_shard_names:
        assert (state_shards[f"{shard_name}_shard"] == model.get_shard(shard_name)).all()


@pytest.mark.depends(on=["test_load_pretrained_distributed_checkpoint"])
def test_load_converted_distributed_checkpoint():
    config_ref = TEST_MODEL_CONFIG_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=_CKPT_PATH,
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )

    model = TEST_MODEL_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=CONVERT_PATH / "distributed_0",
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    config_alt = TEST_MODEL_CONFIG_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=CONVERT_PATH / "distributed_1",
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    _compare_architectures(config_ref, model.config)
    _compare_model_configs(model.config, config_alt)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._distributed.device)
    )[WEIGHT_SHARD_SAVE_NAME]
    assert (weight_shard == model.get_shard(ShardName.weights)).all()


@pytest.mark.depends(on=["test_converted_fast_llm", "test_load_pretrained_distributed_checkpoint"])
def test_load_converted_fast_llm_checkpoint():
    config_ref = TEST_MODEL_CONFIG_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=_CKPT_PATH,
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    model = TEST_MODEL_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=CONVERT_PATH / "fast_llm_0",
            format=FastLLMCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    config_alt = TEST_MODEL_CONFIG_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=CONVERT_PATH / "fast_llm_1",
            format=FastLLMCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    _compare_architectures(config_ref, model.config)
    _compare_architectures(config_ref, config_alt)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._distributed.device)
    )[WEIGHT_SHARD_SAVE_NAME]
    assert (weight_shard == model.get_shard(ShardName.weights)).all()


@pytest.mark.depends(on=["test_converted_fast_llm", "test_load_pretrained_distributed_checkpoint"])
def test_load_converted_huggingface_checkpoint():
    config_ref = TEST_MODEL_CONFIG_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=_CKPT_PATH,
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    model = TEST_MODEL_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=CONVERT_PATH / "huggingface_1",
            format=HUGGINGFACE_CHECKPOINT_FORMAT,
            load_config=ModelConfigType.model,
        ),
        mode=StageMode.weights,
    )
    config_alt = TEST_MODEL_CONFIG_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=CONVERT_PATH / "huggingface_0",
            format=HUGGINGFACE_CHECKPOINT_FORMAT,
            load_config=ModelConfigType.model,
        )
    )
    _compare_architectures(config_ref, model.config)
    _compare_model_configs(model.config, config_alt)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._distributed.device)
    )[WEIGHT_SHARD_SAVE_NAME]
    assert (weight_shard == model.get_shard(ShardName.weights)).all()


@pytest.mark.depends(on=["test_load_converted_fast_llm_checkpoint", "test_load_converted_huggingface_checkpoint"])
def test_run_converted_model():
    model_ref = TEST_MODEL_HF_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=_CKPT_PATH,
            format=DistributedCheckpointFormat,
            load_config=ModelConfigType.model,
        )
    )
    test_input = torch.randint(
        0, model_ref.config.fast_llm_config.base_model.vocab_size, size=(4, 100), dtype=torch.int64, device="cuda"
    )
    output_ref = model_ref(test_input)
    model_from_fast_llm = TEST_MODEL_HF_CLS.from_pretrained(CONVERT_PATH / "fast_llm_0")
    model_from_hf = TEST_MODEL_HF_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=CONVERT_PATH / "huggingface_0",
            format=HUGGINGFACE_CHECKPOINT_FORMAT,
            load_config=ModelConfigType.model,
        )
    )
    errors = []
    compare = CompareConfig()
    model_as_hf = transformers.AutoModelForCausalLM.from_pretrained(
        CONVERT_PATH / "huggingface_0", trust_remote_code=HUGGINGFACE_CHECKPOINT_FORMAT.trust_remote_code
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


@pytest.mark.slow
@pytest.mark.depends(on=["test_load_converted_distributed_checkpoint"])
def test_load_pretrained_distributed_in_dp2(run_test_script):
    run_test_script(
        f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={CONVERT_PATH / 'distributed_0'}",
            f"pretrained.format={DistributedCheckpointFormat.name}",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )


@pytest.mark.depends(on=["test_load_converted_distributed_checkpoint"])
def test_load_pretrained_distributed_with_config(run_test_script):
    run_test_script(
        f"test_{TEST_MODEL}_load_pretrained_distributed_with_config",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={CONVERT_PATH / 'distributed_0'}",
            f"pretrained.format={DistributedCheckpointFormat.name}",
            "schedule.skip_step=True",
        ],
    )


@pytest.mark.depends(on=["test_load_pretrained_distributed_in_dp2"])
def test_load_pretrained_in_dp2_match_checkpoint():
    test_ckpt_path = TEST_RESULTS_PATH / f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2" / "checkpoint" / "1"
    pretrained_config_ref = CheckpointLoadConfig(
        path=_CKPT_PATH,
        format=DistributedCheckpointFormat,
        load_config=ModelConfigType.fast_llm,
    )
    pretrained_config_test = CheckpointLoadConfig(
        path=test_ckpt_path,
        format=DistributedCheckpointFormat,
        load_config=ModelConfigType.fast_llm,
    )
    config_ref = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_ref)
    config_test = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_test)
    _compare_model_configs(config_ref, config_test)
    shards_ref = safetensors.torch.load_file(_CKPT_PATH / "rank_0.safetensors")
    shards_test = [safetensors.torch.load_file(test_ckpt_path / f"rank_{i}.safetensors") for i in range(2)]
    ref_model = TEST_MODEL_CLS(config_ref)
    test_model = TEST_MODEL_CLS(config_test)

    weight_shard_ref_split = shards_ref[WEIGHT_SHARD_SAVE_NAME].split(ref_model._stage_weight_shard_sizes)
    weight_shards_test_split = [
        shard_test[WEIGHT_SHARD_SAVE_NAME].split(test_model._stage_weight_shard_sizes) for shard_test in shards_test
    ]
    for shard_test in shards_test:
        for shard_name, shard in shard_test.items():
            if shard_name != WEIGHT_SHARD_SAVE_NAME:
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


@pytest.mark.slow
@pytest.mark.depends(on=["test_load_pretrained_in_dp2_match_checkpoint"])
def test_load_distributed_checkpoint_dp2():
    # This also tests conversion which uses `FastLLMModel.from_checkpoint`
    pretrained_config_ref = CheckpointLoadConfig(
        path=_CKPT_PATH,
        format=DistributedCheckpointFormat,
        load_config=ModelConfigType.fast_llm,
    )
    pretrained_config_test = CheckpointLoadConfig(
        path=TEST_RESULTS_PATH / f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2" / "checkpoint" / "1",
        format=DistributedCheckpointFormat,
        load_config=ModelConfigType.model,
    )
    config = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_ref)
    model = TEST_MODEL_CLS.from_pretrained(pretrained_config_test, mode=StageMode.weights)
    _compare_model_configs(config, model.config)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._distributed.device)
    )[WEIGHT_SHARD_SAVE_NAME]
    assert (weight_shard == model.get_shard(ShardName.weights)).all()


@pytest.mark.slow
@pytest.mark.depends(on=["test_load_converted_fast_llm_checkpoint", "test_load_pretrained_in_dp2_match_checkpoint"])
def test_load_pretrained_fast_llm_in_dp2(run_test_script):
    run_test_script(
        f"test_{TEST_MODEL}_load_pretrained_fast_llm_in_dp2",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={CONVERT_PATH / 'fast_llm_0'}",
            f"pretrained.format=fast_llm",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )
    for rank in range(2):
        ref_shard = safetensors.torch.load_file(
            TEST_RESULTS_PATH
            / f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2"
            / "checkpoint"
            / "1"
            / f"rank_{rank}.safetensors"
        )
        test_shard = safetensors.torch.load_file(
            TEST_RESULTS_PATH
            / f"test_{TEST_MODEL}_load_pretrained_fast_llm_in_dp2"
            / "checkpoint"
            / "1"
            / f"rank_{rank}.safetensors"
        )
        for name in set(ref_shard) | set(test_shard):
            assert (ref_shard[name] == test_shard[name]).all()


@pytest.mark.slow
@pytest.mark.depends(on=["test_load_converted_huggingface_checkpoint", "test_load_pretrained_in_dp2_match_checkpoint"])
def test_load_pretrained_huggingface_in_dp2(run_test_script):
    run_test_script(
        f"test_{TEST_MODEL}_load_pretrained_huggingface_in_dp2",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={CONVERT_PATH / 'huggingface_0'}",
            f"pretrained.format={HUGGINGFACE_CHECKPOINT_FORMAT.name}",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )
    for rank in range(2):
        ref_shard = safetensors.torch.load_file(
            TEST_RESULTS_PATH
            / f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2"
            / "checkpoint"
            / "1"
            / f"rank_{rank}.safetensors"
        )
        test_shard = safetensors.torch.load_file(
            TEST_RESULTS_PATH
            / f"test_{TEST_MODEL}_load_pretrained_huggingface_in_dp2"
            / "checkpoint"
            / "1"
            / f"rank_{rank}.safetensors"
        )
        for name in set(ref_shard) | set(test_shard):
            assert (ref_shard[name] == test_shard[name]).all()
