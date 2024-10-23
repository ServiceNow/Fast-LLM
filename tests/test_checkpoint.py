import pathlib
import shutil

import pytest
import safetensors.torch
import torch
import transformers
import yaml

from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointLoadConfig, ModelConfigType
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.models.auto import model_registry
from fast_llm.tools.convert import ConversionConfig
from tests.common import (
    CONFIG_COMMON,
    FORCE_REUSE_RESULTS,
    HUGGINGFACE_MODEL_TYPE,
    REUSE_RESULTS,
    TEST_MODEL,
    TEST_MODEL_TYPE,
    TEST_RESULTS_PATH,
    requires_cuda,
    run_test_script,
)
from tests.compare_tensor_logs import CompareConfig, compare_logged_tensor

TEST_MODEL_CONFIG_CLS = model_registry[TEST_MODEL_TYPE]
TEST_MODEL_HF_CLS = TEST_MODEL_CONFIG_CLS.get_huggingface_model_class()
TEST_MODEL_CLS = TEST_MODEL_CONFIG_CLS.get_model_class()
TEST_BASE_MODEL_CONFIG_CLS = TEST_MODEL_CONFIG_CLS.get_base_model_config_cls()
TEST_ARCHITECTURE_CONFIG_CLS = TEST_BASE_MODEL_CONFIG_CLS.architecture_cls


@requires_cuda
@pytest.mark.depends()
def test_checkpoint_and_eval():
    # A baseline config (single-gpu, bf16, flash-attn).
    run_test_script(
        f"test_{TEST_MODEL}_checkpoint_and_eval",
        CONFIG_COMMON
        + ["training.checkpoint.interval=1", "training.validation.interval=2", "training.validation.iterations=1"],
    )


def _prepare_resume_fn(test_path: pathlib.Path, compare_path: pathlib.Path, skip: bool) -> bool:
    if skip and (test_path / "checkpoints" / "2" / "ok").is_file():
        return True
    elif test_path.is_dir():
        shutil.rmtree(test_path)
    shutil.copytree(compare_path, test_path)
    shutil.rmtree(test_path / "checkpoints" / "2")
    assert (test_path / "checkpoints" / "1" / "ok").is_file()
    # TODO: Eval
    shutil.rmtree(test_path / "runs")
    return False


def _compare_resume_fn(test_path: pathlib.Path, compare_path: pathlib.Path):
    for artifact in ["init", "train_1"]:
        path = f"runs/0/artifacts/0/tensor_logs_{artifact}.pt"
        if not (test_path / path).is_file():
            shutil.copy(compare_path / path, test_path / path)


@pytest.mark.depends(on=["test_checkpoint_and_eval"])
def test_resume():
    run_test_script(
        f"test_{TEST_MODEL}_resume",
        CONFIG_COMMON
        + ["training.checkpoint.interval=1", "training.validation.interval=2", "training.validation.iterations=1"],
        compare=f"test_{TEST_MODEL}_checkpoint_and_eval",
        prepare_fn=_prepare_resume_fn,
        compare_fn=_compare_resume_fn,
    )


def _run_conversion(config: ConversionConfig):
    if config.output_path.is_dir() and not REUSE_RESULTS:
        shutil.rmtree(config.output_path)
    if not config.output_path.is_dir():
        if FORCE_REUSE_RESULTS:
            raise RuntimeError(config.output_path)
        config.run(TEST_MODEL_CONFIG_CLS)


_CKPT_PATH = TEST_RESULTS_PATH / f"test_{TEST_MODEL}_checkpoint_and_eval" / "checkpoints" / "2"
_CONVERT_PATH = TEST_RESULTS_PATH / f"test_{TEST_MODEL}_convert_model"


@pytest.mark.depends(on=["test_checkpoint_and_eval"])
def test_convert_distributed_to_state_dict():
    _run_conversion(
        ConversionConfig(
            input_type=CheckpointFormat.distributed,
            input_path=_CKPT_PATH,
            output_type=CheckpointFormat.state_dict,
            output_path=_CONVERT_PATH / "state_dict_0",
        )
    )


@pytest.mark.depends(on=["test_convert_distributed_to_state_dict"])
def test_convert_state_dict_to_huggingface():
    if HUGGINGFACE_MODEL_TYPE is None:
        pytest.skip(f"Conversion not supported for {TEST_MODEL}")
    _run_conversion(
        ConversionConfig(
            input_type=CheckpointFormat.state_dict,
            input_path=_CONVERT_PATH / "state_dict_0",
            output_type=CheckpointFormat.external,
            output_path=_CONVERT_PATH / "huggingface_0",
            model_type=HUGGINGFACE_MODEL_TYPE,
        )
    )


@pytest.mark.depends(on=["test_convert_state_dict_to_huggingface"])
def test_convert_huggingface_to_distributed():
    _run_conversion(
        ConversionConfig(
            input_type=CheckpointFormat.external,
            input_path=_CONVERT_PATH / "huggingface_0",
            output_type=CheckpointFormat.distributed,
            output_path=_CONVERT_PATH / "distributed_0",
        )
    )


@pytest.mark.depends(on=["test_checkpoint_and_eval"])
def test_convert_distributed_to_huggingface():
    if HUGGINGFACE_MODEL_TYPE is None:
        pytest.skip(f"Conversion not supported for {TEST_MODEL}")
    _run_conversion(
        ConversionConfig(
            input_type=CheckpointFormat.distributed,
            input_path=_CKPT_PATH,
            output_type=CheckpointFormat.external,
            output_path=_CONVERT_PATH / "huggingface_1",
            model_type=HUGGINGFACE_MODEL_TYPE,
        )
    )


@pytest.mark.depends(on=["test_convert_distributed_to_huggingface"])
def test_convert_huggingface_to_state_dict():
    _run_conversion(
        ConversionConfig(
            input_type=CheckpointFormat.external,
            input_path=_CONVERT_PATH / "huggingface_1",
            output_type=CheckpointFormat.state_dict,
            output_path=_CONVERT_PATH / "state_dict_1",
        )
    )


@pytest.mark.depends(on=["test_convert_huggingface_to_state_dict"])
def test_convert_state_dict_to_distributed():
    _run_conversion(
        ConversionConfig(
            input_type=CheckpointFormat.state_dict,
            input_path=_CONVERT_PATH / "state_dict_1",
            output_type=CheckpointFormat.distributed,
            output_path=_CONVERT_PATH / "distributed_1",
        )
    )


@pytest.mark.depends(on=["test_convert_huggingface_to_distributed", "test_convert_state_dict_to_distributed"])
def test_converted_distributed():
    # Compare the fast llm weights
    # TODO: Compare configs
    w = safetensors.torch.load_file(_CKPT_PATH / "rank_0.safetensors")
    w0 = safetensors.torch.load_file(_CONVERT_PATH / "distributed_0" / "rank_0.safetensors")
    w1 = safetensors.torch.load_file(_CONVERT_PATH / "distributed_1" / "rank_0.safetensors")
    assert w.keys() == w0.keys() == w1.keys() == {"state_shard"}
    for key in w:
        assert w[key][:1].shape == w0[key].shape, (key, w[key][:1].shape, w0[key].shape)
        assert (w[key][:1] == w0[key]).all(), (w[key][:1], w0[key])
        assert w[key][:1].shape == w1[key].shape, (key, w[key][:1].shape, w1[key].shape)
        assert (w[key][:1] == w1[key]).all(), (w[key][:1], w1[key])


@pytest.mark.depends(on=["test_convert_distributed_to_state_dict", "test_convert_huggingface_to_state_dict"])
def test_converted_state_dict():
    s0 = safetensors.torch.load_file(_CONVERT_PATH / "state_dict_0" / "state_dict_0.safetensors")
    s1 = safetensors.torch.load_file(_CONVERT_PATH / "state_dict_1" / "state_dict_0.safetensors")
    assert s0.keys() == s1.keys()
    for key in s0:
        assert s0[key].shape == s1[key].shape, (key, s0[key].shape, s1[key].shape)
        assert (s0[key] == s1[key]).all(), (key, s0, s1)


@pytest.mark.depends(on=["test_convert_state_dict_to_huggingface", "test_convert_distributed_to_huggingface"])
def test_converted_huggingface():
    h0 = safetensors.torch.load_file(_CONVERT_PATH / "huggingface_0" / "model_0.safetensors")
    h1 = safetensors.torch.load_file(_CONVERT_PATH / "huggingface_1" / "model_0.safetensors")
    assert h0.keys() == h1.keys()
    for key in h0:
        assert h0[key].shape == h1[key].shape, (key, h0[key].shape, h1[key].shape)
        assert (h0[key] == h1[key]).all()


def _compare_configs(config_ref, config_test):
    config_ref.compare(config_test)


@pytest.mark.depends(on=["test_converted_distributed"])
def test_load_pretrained_distributed_checkpoint():
    config = TEST_ARCHITECTURE_CONFIG_CLS.from_dict(
        yaml.safe_load((_CKPT_PATH / ".." / ".." / "config.yaml").open("r")), strict=False
    )
    pretrained_config_ref = CheckpointLoadConfig(
        path=_CKPT_PATH,
        format=CheckpointFormat.distributed,
        optimizer_state=True,
        load_config=ModelConfigType.fast_llm,
    )
    model = TEST_MODEL_CLS.from_pretrained(pretrained_config_ref)
    _compare_configs(config, model._base_model_config)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._state_shard.device)
    )["state_shard"]
    assert (weight_shard == model._state_shard).all()


@pytest.mark.depends(on=["test_load_pretrained_distributed_checkpoint"])
def test_load_converted_distributed_checkpoint():
    pretrained_config_ref = CheckpointLoadConfig(path=_CKPT_PATH, format=CheckpointFormat.distributed)
    pretrained_config_0 = CheckpointLoadConfig(
        path=_CONVERT_PATH / "distributed_0",
        format=CheckpointFormat.distributed,
    )
    pretrained_config_1 = CheckpointLoadConfig(
        path=_CONVERT_PATH / "distributed_1",
        format=CheckpointFormat.distributed,
    )
    config = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_ref)
    model = TEST_MODEL_CLS.from_pretrained(pretrained_config_0)
    config_1 = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_1)
    _compare_configs(config.base_model, model._base_model_config)
    _compare_configs(config.base_model, config_1.base_model)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._state_shard.device)
    )["state_shard"][:1]
    assert (weight_shard == model._state_shard).all()


@pytest.mark.depends(on=["test_converted_state_dict", "test_load_pretrained_distributed_checkpoint"])
def test_load_converted_state_dict_checkpoint():
    pretrained_config_ref = CheckpointLoadConfig(path=_CKPT_PATH, format=CheckpointFormat.distributed)
    pretrained_config_0 = CheckpointLoadConfig(path=_CONVERT_PATH / "state_dict_0", format=CheckpointFormat.state_dict)
    pretrained_config_1 = CheckpointLoadConfig(path=_CONVERT_PATH / "state_dict_1", format=CheckpointFormat.state_dict)
    config = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_ref)
    model = TEST_MODEL_CLS.from_pretrained(pretrained_config_0)
    config_1 = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_1)
    _compare_configs(config.base_model, model._base_model_config)
    _compare_configs(config.base_model, config_1.base_model)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._state_shard.device)
    )["state_shard"][:1]
    assert (weight_shard == model._state_shard).all()


@pytest.mark.depends(on=["test_converted_state_dict", "test_load_pretrained_distributed_checkpoint"])
def test_load_converted_huggingface_checkpoint():
    pretrained_config_ref = CheckpointLoadConfig(
        path=_CKPT_PATH,
        format=CheckpointFormat.distributed,
    )
    pretrained_config_0 = CheckpointLoadConfig(
        path=_CONVERT_PATH / "huggingface_0",
        format=CheckpointFormat.external,
    )
    pretrained_config_1 = CheckpointLoadConfig(
        path=_CONVERT_PATH / "huggingface_1",
        format=CheckpointFormat.external,
    )
    config = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_ref)
    model = TEST_MODEL_CLS.from_pretrained(pretrained_config_0, mode=StageMode.weights)
    config_1 = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_1)
    _compare_configs(config.base_model, model._base_model_config)
    _compare_configs(config.base_model, config_1.base_model)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._state_shard.device)
    )["state_shard"][:1]
    assert (weight_shard == model._state_shard).all()


@pytest.mark.depends(on=["test_load_converted_state_dict_checkpoint", "test_load_converted_huggingface_checkpoint"])
def test_run_converted_model():
    model_ref = TEST_MODEL_HF_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=_CKPT_PATH,
            format=CheckpointFormat.distributed,
        )
    )
    test_input = torch.randint(
        0, model_ref.config.fast_llm_config.base_model.vocab_size, size=(4, 100), dtype=torch.int64, device="cuda"
    )
    output_ref = model_ref(test_input)
    model_from_state_dict = TEST_MODEL_HF_CLS.from_pretrained(_CONVERT_PATH / "state_dict_0")
    model_from_hf = TEST_MODEL_HF_CLS.from_pretrained(
        CheckpointLoadConfig(
            path=_CONVERT_PATH / "huggingface_0",
            format=CheckpointFormat.external,
        )
    )
    errors = []
    compare = CompareConfig()
    model_as_hf = transformers.AutoModelForCausalLM.from_pretrained(_CONVERT_PATH / "huggingface_0").cuda()
    for name, model in zip(
        ("From state dict", "From Huggingface", "Native Huggingface"),
        (model_from_state_dict, model_from_hf, model_as_hf),
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


@pytest.mark.depends(on=["test_load_converted_distributed_checkpoint"])
def test_load_pretrained_distributed_in_dp2():
    run_test_script(
        f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={_CONVERT_PATH / 'distributed_0'}",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )


@pytest.mark.depends(on=["test_load_converted_distributed_checkpoint"])
def test_load_pretrained_distributed_with_config():
    run_test_script(
        f"test_{TEST_MODEL}_load_pretrained_distributed_with_config",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={_CONVERT_PATH / 'distributed_0'}",
            "schedule.skip_step=True",
        ],
    )


@pytest.mark.depends(on=["test_load_pretrained_distributed_in_dp2"])
def test_load_pretrained_in_dp2_match_checkpoint():
    test_ckpt_path = TEST_RESULTS_PATH / f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2" / "checkpoints" / "1"
    pretrained_config_ref = CheckpointLoadConfig(
        path=_CKPT_PATH,
        format=CheckpointFormat.distributed,
        load_config=ModelConfigType.fast_llm,
    )
    pretrained_config_test = CheckpointLoadConfig(
        path=test_ckpt_path,
        format=CheckpointFormat.distributed,
        load_config=ModelConfigType.fast_llm,
    )
    config_ref = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_ref)
    config_test = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_test)
    _compare_configs(config_ref.base_model, config_test.base_model)
    shard_ref = safetensors.torch.load_file(_CKPT_PATH / "rank_0.safetensors")["state_shard"]
    shards_test = [
        safetensors.torch.load_file(test_ckpt_path / f"rank_{i}.safetensors")["state_shard"] for i in range(2)
    ]
    ref_model = TEST_MODEL_CLS(config_ref)
    test_model = TEST_MODEL_CLS(config_test)

    weight_shard_ref_split = shard_ref[0].split(ref_model._stage_shard_sizes)
    weight_shards_test_split = [shard_test[0].split(test_model._stage_shard_sizes) for shard_test in shards_test]
    for shard_test in shards_test:
        assert (shard_test[1:] == 0).all()  # noqa

    assert len(ref_model._stage_shard_sizes) == len(test_model._stage_shard_sizes)
    for i, stage_shard_ref in enumerate(weight_shard_ref_split):
        assert (
            test_model._stage_shard_sizes[i]
            == ref_model._stage_shard_sizes[i] // 2 + (-ref_model._stage_shard_sizes[i] // 2) % 32
        )

        stage_shard_test = torch.concatenate(
            [weight_shard_test_split[i] for weight_shard_test_split in weight_shards_test_split]
        )
        assert (stage_shard_test[: stage_shard_ref.numel()] == stage_shard_ref).all()
        assert (stage_shard_test[stage_shard_ref.numel() :] == 0).all()  # noqa


@pytest.mark.depends(on=["test_load_pretrained_in_dp2_match_checkpoint"])
def test_load_distributed_checkpoint_dp2():
    # This also tests conversion which uses `FastLLMModel.from_checkpoint`
    pretrained_config_ref = CheckpointLoadConfig(
        path=_CKPT_PATH,
        format=CheckpointFormat.distributed,
        load_config=ModelConfigType.fast_llm,
    )
    pretrained_config_test = CheckpointLoadConfig(
        path=TEST_RESULTS_PATH / f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2" / "checkpoints" / "1",
        format=CheckpointFormat.distributed,
    )
    config = TEST_MODEL_CONFIG_CLS.from_pretrained(pretrained_config_ref)
    model = TEST_MODEL_CLS.from_pretrained(pretrained_config_test, mode=StageMode.weights)
    _compare_configs(config.base_model, model._base_model_config)
    weight_shard = safetensors.torch.load_file(
        _CKPT_PATH / "rank_0.safetensors", device=str(model._state_shard.device)
    )["state_shard"][:1]
    assert (weight_shard == model._state_shard).all()


@pytest.mark.depends(on=["test_load_converted_state_dict_checkpoint", "test_load_pretrained_in_dp2_match_checkpoint"])
def test_load_pretrained_state_dict_in_dp2():
    run_test_script(
        f"test_{TEST_MODEL}_load_pretrained_state_dict_in_dp2",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={_CONVERT_PATH / 'state_dict_0'}",
            f"pretrained.format=state_dict",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )
    for rank in range(2):
        ref_shard = safetensors.torch.load_file(
            TEST_RESULTS_PATH
            / f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2"
            / "checkpoints"
            / "1"
            / f"rank_{rank}.safetensors"
        )["state_shard"]
        test_shard = safetensors.torch.load_file(
            TEST_RESULTS_PATH
            / f"test_{TEST_MODEL}_load_pretrained_state_dict_in_dp2"
            / "checkpoints"
            / "1"
            / f"rank_{rank}.safetensors"
        )["state_shard"]
        assert (ref_shard == test_shard).all()


@pytest.mark.depends(on=["test_load_converted_huggingface_checkpoint", "test_load_pretrained_in_dp2_match_checkpoint"])
def test_load_pretrained_huggingface_in_dp2():
    run_test_script(
        f"test_{TEST_MODEL}_load_pretrained_huggingface_in_dp2",
        CONFIG_COMMON
        + [
            "training.checkpoint.interval=1",
            "training.train_iters=1",
            f"pretrained.path={_CONVERT_PATH / 'huggingface_0'}",
            f"pretrained.format=external",
            "schedule.skip_step=True",
        ],
        num_gpus=2,
    )
    for rank in range(2):
        ref_shard = safetensors.torch.load_file(
            TEST_RESULTS_PATH
            / f"test_{TEST_MODEL}_load_pretrained_distributed_in_dp2"
            / "checkpoints"
            / "1"
            / f"rank_{rank}.safetensors"
        )["state_shard"]
        test_shard = safetensors.torch.load_file(
            TEST_RESULTS_PATH
            / f"test_{TEST_MODEL}_load_pretrained_huggingface_in_dp2"
            / "checkpoints"
            / "1"
            / f"rank_{rank}.safetensors"
        )["state_shard"]
        assert (ref_shard == test_shard).all()
