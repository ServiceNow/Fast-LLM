import logging
import os
import pathlib
import typing

from fast_llm.data.data.abstract import Data
from fast_llm.engine.evaluation.config import LmEvalEvaluatorConfig
from fast_llm.engine.evaluation.evaluator import Evaluator
from fast_llm.engine.evaluation.lm_eval.fast_llm_wrapper import FastLLMLmEvalWrapper
from fast_llm.engine.inference.huggingface import HuggingfacePreTrainedModel
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.runner import ScheduleRunner

logger = logging.getLogger(__name__)


class LmEvalEvaluator[ConfigType: LmEvalEvaluatorConfig](Evaluator[ConfigType]):
    _hf_model: HuggingfacePreTrainedModel
    _flm_wrapper: FastLLMLmEvalWrapper

    def setup(
        self,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        run_count: int,
    ) -> None:
        if "HUGGINGFACE_API_KEY_PATH" in os.environ:
            os.environ["HF_TOKEN"] = pathlib.Path(os.environ["HUGGINGFACE_API_KEY_PATH"]).read_text().strip()
        else:
            if not "HF_TOKEN" in os.environ:
                logger.warning(
                    "No `HF_TOKEN` or `HUGGINGFACE_API_KEY_PATH` environment variable provided. "
                    "Assuming the user has already logged in to the Hugging Face Hub."
                )

        from fast_llm.engine.evaluation.lm_eval.fast_llm_wrapper import FastLLMLmEvalWrapper

        super().setup(multi_stage, runner, data, run_count)

        hf_model = multi_stage.config_class.get_huggingface_model_for_causal_lm_class()(multi_stage, runner=runner)

        # For reporting purposes, just to indicate it is from Fast-LLM
        # as lm_eval.simple_evaluate will take it for results['config']['model']
        hf_model.config.name_or_path = type(hf_model).__name__

        self._flm_wrapper = FastLLMLmEvalWrapper(
            model=hf_model,
            tokenizer=self._config.tokenizer.get_tokenizer(),
            truncation=self._config.truncation,
            logits_cache=self._config.logits_cache,
            add_bos_token=self._config.add_bos_token,
            prefix_token_id=self._config.prefix_token_id,
            max_length=self._config.max_length,
            communication_timeout_sec=self._config.communication_timeout_sec,
        )
        self._is_setup = True

    def run(
        self,
        run_index: int | None,
        metrics: dict[str, typing.Any],
    ) -> None:
        assert self._is_setup
        self._flm_wrapper.run(self._config.cli_args, metrics.get("completed_steps", 0), run_index)
