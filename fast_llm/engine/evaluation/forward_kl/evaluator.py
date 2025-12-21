import logging
import typing

import torch
import torch.nn.functional as F

from fast_llm.config import NoAutoValidate
from fast_llm.core.distributed import all_reduce, safe_barrier
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
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.models.gpt.config import GPTBatchConfig

logger = logging.getLogger(__name__)


class ForwardKLEvaluator[ConfigType: ForwardKLEvaluatorConfig](Evaluator[ConfigType]):
    _inference_runner: InferenceRunner
    _sequence_length: int
    _micro_sequence_length: int

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

        # Get sequence configuration from training batch config (required for SP support)
        self._sequence_length = self._batch_config.sequence_length
        self._micro_sequence_length = self._batch_config.micro_sequence_length

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

        # Shard traces across data-parallel ranks
        data_rank = self._distributed.config.data_rank
        data_parallel = self._distributed.config.data_parallel

        traces = datasets.load_dataset(
            self._config.dataset_path,
            name=self._config.task,
            split="validation",
            trust_remote_code=self._config.trust_remote_code,
        )

        # Apply num_samples limit before sharding to preserve semantics
        # (num_samples = total traces across all ranks, not per-rank)
        if self._config.num_samples and len(traces) > self._config.num_samples:
            traces = traces.select(range(self._config.num_samples))

        # Shard across DP ranks (lazy operation - just changes which indices are accessible)
        traces = traces.shard(num_shards=data_parallel, index=data_rank)

        total_kl = 0.0
        num_traces = 0
        num_skipped = 0

        # Collect traces for this rank, filtering by length
        rank_traces = []
        for trace in traces:
            trace_len = len(trace["prompt_tokens"]) + len(trace["completion_tokens"])
            if trace_len > self._sequence_length:
                num_skipped += 1
                continue
            rank_traces.append(trace)

        if num_skipped > 0:
            logger.warning(
                f"Skipped {num_skipped} traces exceeding sequence length {self._sequence_length}"
            )

        # Process traces in batches
        for i in range(0, len(rank_traces), self._config.batch_size):
            batch = rank_traces[i : i + self._config.batch_size]

            student_log_probs = self._compute_batch_log_probs(batch)

            # student_log_probs is None on non-last pipeline ranks (they don't have logits)
            if student_log_probs is not None:
                for j, trace in enumerate(batch):
                    total_kl += trace["teacher_log_prob"] - student_log_probs[j]
                    num_traces += 1

            torch.cuda.empty_cache()

        # Reduce across data group (sum KL and counts from all DP ranks)
        if self._distributed.data_group:
            total_kl_tensor = torch.tensor([total_kl], dtype=torch.float64, device=self._distributed.device)
            num_traces_tensor = torch.tensor([num_traces], dtype=torch.int64, device=self._distributed.device)
            num_skipped_tensor = torch.tensor([num_skipped], dtype=torch.int64, device=self._distributed.device)
            all_reduce(total_kl_tensor, group=self._distributed.data_group)
            all_reduce(num_traces_tensor, group=self._distributed.data_group)
            all_reduce(num_skipped_tensor, group=self._distributed.data_group)
            total_kl = total_kl_tensor.item()
            num_traces = int(num_traces_tensor.item())
            num_skipped = int(num_skipped_tensor.item())

        # Reduce across pipeline group (last PP rank has the values, others have zeros)
        if self._distributed.pipeline_group:
            total_kl_tensor = torch.tensor([total_kl], dtype=torch.float64, device=self._distributed.device)
            num_traces_tensor = torch.tensor([num_traces], dtype=torch.int64, device=self._distributed.device)
            all_reduce(total_kl_tensor, group=self._distributed.pipeline_group)
            all_reduce(num_traces_tensor, group=self._distributed.pipeline_group)
            total_kl = total_kl_tensor.item()
            num_traces = int(num_traces_tensor.item())

        return total_kl / num_traces if num_traces > 0 else 0.0, num_traces, num_skipped

    def _compute_batch_log_probs(self, batch: list[dict[str, typing.Any]]) -> list[float] | None:
        samples = []
        prompt_lengths = []
        completion_lengths = []

        for trace in batch:
            prompt = trace["prompt_tokens"]
            completion = trace["completion_tokens"]
            full = prompt + completion
            actual_len = len(full)
            # Pad to training sequence length (required for SP support)
            pad_len = self._sequence_length - actual_len

            tokens = torch.tensor(full + [0] * pad_len, dtype=torch.int64)
            samples.append(LanguageModelSample(TokenSample(tokens, lengths=[actual_len])))
            prompt_lengths.append(len(prompt))
            completion_lengths.append(len(completion))

        lm_batch = LanguageModelBatch.from_samples(samples)

        # Create batch config with training's sequence settings (required for SP support)
        with NoAutoValidate():
            batch_config = GPTBatchConfig(
                micro_batch_size=len(batch),
                sequence_length=self._sequence_length,
                micro_sequence_length=self._micro_sequence_length,
            )
        batch_config.setup(self._distributed.config)
        batch_config.validate()

        # Get preprocessing metadata using GPTBatchConfig (enables proper SP splitting)
        preprocessed_meta = self._multi_stage.base_model.preprocess_meta(batch_config, PhaseType.inference)

        preprocessed = self._multi_stage.base_model.preprocess_batch(
            lm_batch,
            preprocessed_meta,
            phase=PhaseType.inference,
            iteration=0,
        )

        for input_, kwargs in preprocessed:
            kwargs["global_logits"] = True
            self._inference_runner.forward(input_, kwargs)

        # With pipeline parallelism, only the last stage has logits.
        # Other stages participate in the forward pass but don't compute logits.
        if "logits" not in kwargs:
            return None

        logits = kwargs["logits"]

        if kwargs.get(AttentionKwargs.sequence_first, False):
            logits = logits.transpose(0, 1)

        results = []
        device = logits.device
        for idx in range(len(batch)):
            prompt_len = prompt_lengths[idx]
            completion_len = completion_lengths[idx]

            pred_logits = logits[idx, prompt_len - 1 : prompt_len + completion_len - 1]
            targets = lm_batch.tokens.tokens[idx, prompt_len : prompt_len + completion_len].to(device)

            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            results.append(token_log_probs.sum().item())

        return results
