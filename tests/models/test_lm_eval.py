import pytest

from tests.models.test_checkpoint import _prepare_resume_fn
from tests.utils.dataset import TOKENIZER_PATH, download_santacoder_tokenizer
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda

# NOTE: These tests only verify that the functionality runs without crashing.
# NOTE: The tokenizer is from a LLaMA-style model, which may not be suitable for all models,
#       but it should be sufficient since we are not concerned with actual accuracy in this tests.


try:
    import lm_eval  # isort:skip

    _lm_eval_installed = True
except ImportError:
    _lm_eval_installed = False

requires_lm_eval = pytest.mark.skipif(not _lm_eval_installed, reason="lm_eval is not installed")


@pytest.fixture(scope="module")
def santacoder_tokenizer_path():
    download_santacoder_tokenizer()
    return TOKENIZER_PATH


def get_lm_eval_config(base_path, santacoder_tokenizer_path):
    # TODO: Investigate why loading the tokenizer here gives a vocab_size
    #       smaller than 49157, which is the size when loaded by Fast-LLM.
    return [
        f"data.tokenizer.path={santacoder_tokenizer_path}",
        f"model.base_model.vocab_size=49157",
        "training.evaluators.evaluation_test.interval=1",
        "training.evaluators.evaluation_test.evaluator.type=lm_eval",
        "training.evaluators.evaluation_test.evaluator.cli_args="
        f'["--tasks","gsm8k,xnli_en,wikitext","--output_path","{str(base_path / "lm_eval")}","--limit","10"]',
    ]


@pytest.mark.extra_slow
@requires_lm_eval
@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_lm_eval_in_training(run_test_script_for_all_models, run_test_script_base_path, santacoder_tokenizer_path):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_in_training", santacoder_tokenizer_path)
        + ["training.checkpoint.interval=1"]
    )


@pytest.mark.extra_slow
@requires_lm_eval
@requires_cuda
@pytest.mark.depends_on(on=["test_lm_eval_in_training[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_lm_eval_evaluation(run_test_script_for_all_models, run_test_script_base_path, santacoder_tokenizer_path):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_evaluation", santacoder_tokenizer_path),
        compare="test_lm_eval_in_training",
        prepare_fn=_prepare_resume_fn,
        do_compare=False,
        runnable_type="evaluate",
    )


@pytest.mark.extra_slow
@requires_lm_eval
@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_lm_eval_in_training_dp2(run_test_script_for_all_models, run_test_script_base_path, santacoder_tokenizer_path):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_in_training_dp2", santacoder_tokenizer_path)
        + ["training.checkpoint.interval=1"],
        num_gpus=2,
    )
