import huggingface_hub
import pytest
import transformers

from tests.models.test_checkpoint import _prepare_resume_fn
from tests.utils.model_configs import ModelTestingGroup
from tests.utils.utils import requires_cuda, requires_lm_eval

# NOTE: These tests only verify that the functionality runs without crashing.
# NOTE: The tokenizer is from a LLaMA-style model, which may not be suitable for all models,
#       but it should be sufficient since we are not concerned with actual accuracy in this tests.


@pytest.fixture(scope="module")
def model_path(result_path):
    return huggingface_hub.snapshot_download(
        repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        local_dir=result_path / "lm_eval/model",
    )


def get_lm_eval_config(base_path, tokenizer_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    return [
        f"data.tokenizer.path={tokenizer_path}",
        f"model.base_model.vocab_size={tokenizer.vocab_size}",
        "training.evaluators.evaluation_test.interval=1",
        "training.evaluators.evaluation_test.evaluator.type=lm_eval",
        "training.evaluators.evaluation_test.evaluator.cli_args="
        f'["--tasks","gsm8k,xnli_en,wikitext","--output_path","{str(base_path / "lm_eval")}","--limit","10"]',
    ]


@pytest.mark.extra_slow
@requires_lm_eval
@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_lm_eval_in_training(run_test_script_for_all_models, run_test_script_base_path, model_path):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_in_training", model_path)
        + ["training.checkpoint.interval=1"]
    )


@pytest.mark.extra_slow
@requires_lm_eval
@requires_cuda
@pytest.mark.depends_on(on=["test_lm_eval_in_training[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.basic)
def test_lm_eval_evaluation(run_test_script_for_all_models, run_test_script_base_path, model_path):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_evaluation", model_path),
        compare="test_lm_eval_in_training",
        prepare_fn=_prepare_resume_fn,
        do_compare=False,
        task="evaluate",
    )


@pytest.mark.extra_slow
@requires_lm_eval
@requires_cuda
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_lm_eval_in_training_dp2(run_test_script_for_all_models, run_test_script_base_path, model_path):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_in_training_dp2", model_path)
        + ["training.checkpoint.interval=1"],
        num_gpus=2,
    )


@pytest.mark.extra_slow
@requires_lm_eval
@requires_cuda
@pytest.mark.depends_on(on=["test_lm_eval_in_training_dp2[{model_testing_config}]"])
@pytest.mark.model_testing_group(ModelTestingGroup.distributed)
def test_lm_eval_evaluation_dp2(run_test_script_for_all_models, run_test_script_base_path, model_path):
    run_test_script_for_all_models(
        get_lm_eval_config(run_test_script_base_path / "test_lm_eval_evaluation_dp2", model_path),
        compare="test_lm_eval_in_training_dp2",
        prepare_fn=_prepare_resume_fn,
        do_compare=False,
        num_gpus=2,
        task="evaluate",
    )
