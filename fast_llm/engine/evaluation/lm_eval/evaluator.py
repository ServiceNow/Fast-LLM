import logging
import os
import pathlib
import typing

from fast_llm.data.data.abstract import Data
from fast_llm.engine.config_utils.run import Run
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.evaluation.config import LmEvalEvaluatorConfig
from fast_llm.engine.evaluation.evaluator import (
    EvaluationMetrics,
    Evaluator,
    EvaluatorSamplingParameters,
    TrainingProgress,
)
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.runner import ScheduleRunner

if typing.TYPE_CHECKING:
    from fast_llm.engine.evaluation.lm_eval.fast_llm_wrapper import FastLLMLmEvalWrapper
    from fast_llm.engine.inference.huggingface import HuggingfaceBaseModelForCausalLM

logger = logging.getLogger(__name__)


class LmEvalEvaluator[ConfigType: LmEvalEvaluatorConfig](Evaluator[ConfigType]):
    _hf_model: "HuggingfaceBaseModelForCausalLM" = None
    _flm_wrapper: "FastLLMLmEvalWrapper" = None

    def setup(
        self,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        phase: PhaseType,
    ) -> None:
        if "HUGGINGFACE_API_KEY_PATH" in os.environ:
            os.environ["HF_TOKEN"] = pathlib.Path(os.environ["HUGGINGFACE_API_KEY_PATH"]).open("r").read().strip()
        else:
            if not "HF_TOKEN" in os.environ:
                logger.warning(
                    "No `HF_TOKEN` or `HUGGINGFACE_API_KEY_PATH` environment variable provided. "
                    "Assuming the user has already logged in to the Hugging Face Hub."
                )

        from fast_llm.engine.evaluation.lm_eval.fast_llm_wrapper import FastLLMLmEvalWrapper

        super().setup(distributed, run, multi_stage, runner, data, phase)

        self._hf_model = self._multi_stage.config_class.get_huggingface_model_for_causal_lm_class()(
            self._multi_stage, runner=self._runner
        )

        # For reporting purposes, just to indicate it is from Fast-LLM
        # as lm_eval.simple_evaluate will take it for results['config']['model']
        self._hf_model.config.name_or_path = type(self._hf_model).__name__

        self._flm_wrapper = FastLLMLmEvalWrapper(
            model=self._hf_model,
            tokenizer=self._data.tokenizer.tokenizer,
            truncation=self._config.truncation,
            logits_cache=self._config.logits_cache,
            add_bos_token=self._config.add_bos_token,
            prefix_token_id=self._config.prefix_token_id,
            max_length=self._config.max_length,
        )
        self._is_setup = True

    def run(
        self,
        training_progress: TrainingProgress | None = None,
        run_index: int | None = None,
    ) -> EvaluationMetrics:
        assert self._is_setup

        # completed_steps is added to output_path like output_path/runs/run_index/completed_steps/
        completed_steps = 0 if training_progress is None else training_progress.completed_steps

        self._flm_wrapper.run(self._config.cli_args, completed_steps, self._run.index)

        # lm_eval logs to disc, wandb and prints to screen itself
        return EvaluationMetrics()

    def get_sampling_parameters(self) -> EvaluatorSamplingParameters | None:
        return None
