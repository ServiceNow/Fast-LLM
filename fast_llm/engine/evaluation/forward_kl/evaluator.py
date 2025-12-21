import logging
import typing

import datasets
import torch
import torch.nn.functional as F

from fast_llm.core.distributed import safe_barrier
from fast_llm.data.data.abstract import Data
from fast_llm.engine.config_utils.run import Run, log_main_rank
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.evaluation.evaluator import (
    EvaluationMetrics,
    Evaluator,
    EvaluatorSamplingParameters,
    TrainingProgress,
)
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.runner import ScheduleRunner

if typing.TYPE_CHECKING:
    from fast_llm.engine.evaluation.config import ForwardKLEvaluatorConfig
    from fast_llm.engine.inference.huggingface import HuggingfacePreTrainedModel

logger = logging.getLogger(__name__)


class ForwardKLEvaluator[ConfigType: "ForwardKLEvaluatorConfig"](Evaluator[ConfigType]):
    _hf_model: "HuggingfacePreTrainedModel" = None

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

        self._hf_model = self._multi_stage.config_class.get_huggingface_model_for_causal_lm_class()(
            self._multi_stage, runner=self._runner
        )
        self._is_setup = True

    def get_sampling_parameters(self) -> EvaluatorSamplingParameters | None:
        return None

    def run(
        self,
        training_progress: TrainingProgress | None = None,
        run_index: int | None = None,
    ) -> EvaluationMetrics:
        assert self._is_setup

        safe_barrier(self._distributed.world_group, f"forward_kl_{self._name} begin")

        traces = self._load_traces()
        if len(traces) == 0:
            return EvaluationMetrics()

        forward_kl, num_traces = self._compute_forward_kl(traces)

        safe_barrier(self._distributed.world_group, f"forward_kl_{self._name} end")

        metrics = {
            f"validation.{self._name}": {
                "forward_kl": forward_kl,
                "num_traces": num_traces,
            }
        }

        if training_progress is not None:
            metrics[f"validation.{self._name}"]["iteration"] = training_progress.completed_steps

        formatted = f"Forward KL ({self._name}): {forward_kl:.4f} ({num_traces} traces)"
        log_main_rank(formatted)

        return EvaluationMetrics(metrics, formatted)

    def _load_traces(self) -> datasets.Dataset:
        if self._config.dataset_path is None:
            return []

        return datasets.load_dataset(
            self._config.dataset_path,
            name=self._config.task,
            split="validation",
            trust_remote_code=self._config.trust_remote_code,
        )

    @torch.inference_mode()
    def _compute_forward_kl(self, traces: datasets.Dataset) -> tuple[float, int]:
        device = self._hf_model.device
        total_kl = 0.0
        num_traces = 0

        num_samples = min(len(traces), self._config.num_samples) if self._config.num_samples else len(traces)

        for i in range(0, num_samples, self._config.batch_size):
            batch_end = min(i + self._config.batch_size, num_samples)
            batch = traces.select(range(i, batch_end))

            student_log_probs = self._compute_batch_log_probs(batch, device)

            for j, trace in enumerate(batch):
                teacher_lp = trace["teacher_log_prob"]
                student_lp = student_log_probs[j]
                total_kl += teacher_lp - student_lp
                num_traces += 1

            torch.cuda.empty_cache()

        return total_kl / num_traces if num_traces > 0 else 0.0, num_traces

    def _compute_batch_log_probs(self, batch: datasets.Dataset, device: torch.device) -> list[float]:
        max_len = max(len(t["prompt_tokens"]) + len(t["completion_tokens"]) for t in batch)
        pad_token_id = getattr(self._hf_model.config, "pad_token_id", 0) or 0

        input_ids_list = []
        attention_mask_list = []
        prompt_lengths = []
        completion_lengths = []

        for trace in batch:
            prompt = trace["prompt_tokens"]
            completion = trace["completion_tokens"]
            full = prompt + completion
            padding = [pad_token_id] * (max_len - len(full))

            input_ids_list.append(full + padding)
            attention_mask_list.append([1] * len(full) + [0] * len(padding))
            prompt_lengths.append(len(prompt))
            completion_lengths.append(len(completion))

        input_ids = torch.tensor(input_ids_list, device=device)
        attention_mask = torch.tensor(attention_mask_list, device=device)

        output = self._hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = output.logits

        results = []
        for idx in range(len(batch)):
            prompt_len = prompt_lengths[idx]
            completion_len = completion_lengths[idx]

            pred_logits = logits[idx, prompt_len - 1 : prompt_len + completion_len - 1]
            targets = input_ids[idx, prompt_len : prompt_len + completion_len]

            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            results.append(token_log_probs.sum().item())

        return results
