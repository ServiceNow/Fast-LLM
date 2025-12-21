import logging
import typing

import torch
import torch.nn.functional as F

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.abstract import Data
from fast_llm.data.sample.language_model import LanguageModelBatch, LanguageModelSample
from fast_llm.data.sample.token import TokenSample
from fast_llm.engine.config_utils.run import Run, log_main_rank
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.evaluation.config import ForwardKLEvaluatorConfig
from fast_llm.engine.evaluation.evaluator import (
    EvaluationMetrics,
    Evaluator,
    EvaluatorSamplingParameters,
    TrainingProgress,
)
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.runner import ScheduleRunner

logger = logging.getLogger(__name__)


class ForwardKLEvaluator[ConfigType: ForwardKLEvaluatorConfig](Evaluator[ConfigType]):
    _inference_runner: InferenceRunner
    _max_sequence_length: int

    def setup(
        self,
        distributed: Distributed,
        run: Run,
        multi_stage: FastLLMModel,
        runner: ScheduleRunner,
        data: Data,
        phase: PhaseType,
    ) -> None:
        super().setup(distributed, run, multi_stage, runner, data, phase)
        self._inference_runner = InferenceRunner(self._multi_stage, runner=self._runner)
        self._inference_runner.setup()

        if self._config.max_sequence_length is not None:
            self._max_sequence_length = self._config.max_sequence_length
        else:
            self._max_sequence_length = self._multi_stage.base_model._config.embeddings.num_position_embeddings

        self._is_setup = True

    def get_sampling_parameters(self) -> EvaluatorSamplingParameters | None:
        return None

    def run(
        self,
        training_progress: TrainingProgress | None = None,
        run_index: int | None = None,
    ) -> EvaluationMetrics:
        assert self._is_setup

        if self._config.dataset_path is None:
            return EvaluationMetrics()

        safe_barrier(self._distributed.world_group, f"forward_kl_{self._name} begin")

        forward_kl, num_traces, num_skipped = self._compute_forward_kl()

        safe_barrier(self._distributed.world_group, f"forward_kl_{self._name} end")

        if num_traces == 0:
            return EvaluationMetrics()

        metrics = {
            f"validation.{self._name}": {
                "forward_kl": forward_kl,
                "num_traces": num_traces,
            }
        }

        if training_progress is not None:
            metrics[f"validation.{self._name}"]["iteration"] = training_progress.completed_steps

        formatted = f"Forward KL ({self._name}): {forward_kl:.4f} ({num_traces} traces)"
        if num_skipped > 0:
            formatted += f" [{num_skipped} skipped]"
        log_main_rank(formatted)

        return EvaluationMetrics(metrics, formatted)

    @torch.inference_mode()
    def _compute_forward_kl(self) -> tuple[float, int, int]:
        import datasets

        traces = datasets.load_dataset(
            self._config.dataset_path,
            name=self._config.task,
            split="validation",
            trust_remote_code=self._config.trust_remote_code,
        )

        total_kl = 0.0
        num_traces = 0
        num_skipped = 0
        num_samples = min(len(traces), self._config.num_samples) if self._config.num_samples else len(traces)

        for i in range(0, num_samples, self._config.batch_size):
            batch_indices = range(i, min(i + self._config.batch_size, num_samples))
            batch = []
            for j in batch_indices:
                trace = traces[j]
                trace_len = len(trace["prompt_tokens"]) + len(trace["completion_tokens"])
                if trace_len > self._max_sequence_length:
                    logger.warning(
                        f"Skipping trace {j}: length {trace_len} exceeds max {self._max_sequence_length}"
                    )
                    num_skipped += 1
                    continue
                batch.append(trace)

            if not batch:
                continue

            student_log_probs = self._compute_batch_log_probs(batch)

            for j, trace in enumerate(batch):
                total_kl += trace["teacher_log_prob"] - student_log_probs[j]
                num_traces += 1

            torch.cuda.empty_cache()

        return total_kl / num_traces if num_traces > 0 else 0.0, num_traces, num_skipped

    def _compute_batch_log_probs(self, batch: list[dict[str, typing.Any]]) -> list[float]:
        max_len = max(len(t["prompt_tokens"]) + len(t["completion_tokens"]) for t in batch)

        samples = []
        prompt_lengths = []
        completion_lengths = []

        for trace in batch:
            prompt = trace["prompt_tokens"]
            completion = trace["completion_tokens"]
            full = prompt + completion
            actual_len = len(full)
            pad_len = max_len - actual_len

            tokens = torch.tensor(full + [0] * pad_len, dtype=torch.int64)
            samples.append(LanguageModelSample(TokenSample(tokens, lengths=[actual_len])))
            prompt_lengths.append(len(prompt))
            completion_lengths.append(len(completion))

        lm_batch = LanguageModelBatch.from_samples(samples)

        preprocessed = self._multi_stage.base_model.preprocess_batch(
            lm_batch,
            phase=PhaseType.inference,
            iteration=0,
        )

        for input_, kwargs in preprocessed:
            kwargs["global_logits"] = True
            self._inference_runner.forward(input_, kwargs)
            logits = kwargs["logits"]

        sequence_first = kwargs.get("sequence_first", False)
        if sequence_first:
            logits = logits.transpose(0, 1)

        results = []
        for idx in range(len(batch)):
            prompt_len = prompt_lengths[idx]
            completion_len = completion_lengths[idx]

            pred_logits = logits[idx, prompt_len - 1 : prompt_len + completion_len - 1]
            targets = lm_batch.tokens.tokens[idx, prompt_len : prompt_len + completion_len]

            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            results.append(token_log_probs.sum().item())

        return results
