import pathlib
import shutil

import pytest

from tests.utils.dataset import TOKENIZER_PATH, download_santacoder_tokenizer
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda

# NOTE: These tests only verify that the functionality runs without crashing.
# NOTE: The tokenizer is from a LLaMA-style model, which may not be suitable for all models,
#       but it should be sufficient since we are not concerned with actual accuracy in this tests.


@pytest.fixture(scope="module")
def tokenizer_path():
    download_santacoder_tokenizer()
    return TOKENIZER_PATH


@pytest.fixture(scope="function")
def get_lm_eval_config(tokenizer_path, monkeypatch):
    # TODO: Investigate why loading the tokenizer here gives a vocab_size
    #       smaller than 49157, which is the size when loaded by Fast-LLM.
    import lm_eval.evaluator

    # lm_eval gathers lots of system info when reporting results, and this is extremely slow, so we skip here.
    monkeypatch.setattr(lm_eval.evaluator, "add_env_info", lambda x: None, raising=True)

    def do_get_lm_eval_config(base_path):
        import lm_eval.tasks

        task_dir = pathlib.Path(lm_eval.tasks.__file__).parent.resolve()
        return [
            f"data.tokenizer.path={tokenizer_path}",
            f"model.base_model.vocab_size=49157",
            "training.evaluators.evaluation_test.interval=2",
            "training.evaluators.evaluation_test.evaluator.type=lm_eval",
            "training.evaluators.evaluation_test.evaluator.cli_args="
            f'["--tasks=wikitext",'
            f'"--output_path={str(base_path / "lm_eval")}",'
            # lm_eval loads all available tasks by default which is slow.
            f'"--include_path={str(task_dir / "wikitext")}",'
            f'"--no_defaults",'
            f'"--limit=1",'
            f'"--batch_size=1",'
            f'"--verbosity=DEBUG"]',
        ]

    return do_get_lm_eval_config


# "gsm8k,xnli_en,wikitext"


@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.generate)
def test_lm_eval_in_training(run_test_script_for_all_models, run_test_script_base_path, get_lm_eval_config):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_in_training") + ["training.checkpoint.interval=2"]
    )


def _copy_training_output(test_path: pathlib.Path, compare_path: pathlib.Path):
    shutil.copytree(compare_path, test_path)


@requires_cuda
@pytest.mark.depends_on(on=["test_lm_eval_in_training[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.generate)
def test_lm_eval_evaluation(run_test_script_for_all_models, run_test_script_base_path, get_lm_eval_config):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_in_training"),
        compare="test_lm_eval_in_training",
        prepare_fn=_copy_training_output,
        do_compare=False,
        runnable_type="evaluate",
    )


@requires_cuda
@pytest.mark.depends_on(on=["test_lm_eval_in_training[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.generate, ModelTestingGroup.distributed)
def test_lm_eval_in_training_dp2(run_test_script_for_all_models, run_test_script_base_path, get_lm_eval_config):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_in_training_dp2")
        + ["training.checkpoint.interval=1"],
        num_gpus=2,
    )
