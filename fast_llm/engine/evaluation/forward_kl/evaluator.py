import dataclasses
import gc
import hashlib
import logging

import torch
import torch.nn.functional as F

from fast_llm.config import NoAutoValidate
from fast_llm.core.distributed import allreduce_scalar, safe_barrier
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
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.models.gpt.model import GPTInferenceRunner

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TraceTensors:
    tokens: torch.Tensor  # (num_traces, sequence_length)
    prompt_lens: torch.Tensor  # (num_traces,)
    completion_lens: torch.Tensor  # (num_traces,)
    problem_indices: torch.Tensor  # (num_traces,)
    teacher_log_probs: torch.Tensor  # (num_traces,)
    corrects: torch.Tensor  # (num_traces,)
    num_problems: int
    num_skipped: int

    def __len__(self) -> int:
        return self.tokens.shape[0]

    @classmethod
    def empty(cls, sequence_length: int, device: torch.device, num_skipped: int = 0) -> "TraceTensors":
        return cls(
            tokens=torch.empty((0, sequence_length), dtype=torch.int64, device=device),
            prompt_lens=torch.empty(0, dtype=torch.int64, device=device),
            completion_lens=torch.empty(0, dtype=torch.int64, device=device),
            problem_indices=torch.empty(0, dtype=torch.int64, device=device),
            teacher_log_probs=torch.empty(0, dtype=torch.float64, device=device),
            corrects=torch.empty(0, dtype=torch.bool, device=device),
            num_problems=0,
            num_skipped=num_skipped,
        )

    @classmethod
    def from_traces(
        cls,
        traces: list[dict],
        sequence_length: int,
        device: torch.device,
    ) -> "TraceTensors":
        pid_to_idx: dict[str, int] = {}
        valid_traces: list[tuple[list[int], list[int], str, float, bool]] = []
        num_skipped = 0

        for t in traces:
            prompt, completion = t["prompt_tokens"], t["completion_tokens"]
            if len(prompt) + len(completion) > sequence_length:
                num_skipped += 1
                continue
            valid_traces.append((prompt, completion, t["problem_id"], t["teacher_log_prob"], t["correct"]))

        if not valid_traces:
            return cls.empty(sequence_length, device, num_skipped)

        n = len(valid_traces)
        tokens = torch.zeros((n, sequence_length), dtype=torch.int64, device=device)
        prompt_lens = torch.empty(n, dtype=torch.int64, device=device)
        completion_lens = torch.empty(n, dtype=torch.int64, device=device)
        problem_indices = torch.empty(n, dtype=torch.int64, device=device)
        teacher_log_probs = torch.empty(n, dtype=torch.float64, device=device)
        corrects = torch.empty(n, dtype=torch.bool, device=device)

        for i, (prompt, completion, pid, teacher_lp, correct) in enumerate(valid_traces):
            seq = prompt + completion
            tokens[i, : len(seq)] = torch.tensor(seq, dtype=torch.int64, device=device)
            prompt_lens[i] = len(prompt)
            completion_lens[i] = len(completion)

            if pid not in pid_to_idx:
                pid_to_idx[pid] = len(pid_to_idx)
            problem_indices[i] = pid_to_idx[pid]
            teacher_log_probs[i] = teacher_lp
            corrects[i] = correct

        return cls(
            tokens=tokens,
            prompt_lens=prompt_lens,
            completion_lens=completion_lens,
            problem_indices=problem_indices,
            teacher_log_probs=teacher_log_probs,
            corrects=corrects,
            num_problems=len(pid_to_idx),
            num_skipped=num_skipped,
        )


class ForwardKLEvaluator[ConfigType: ForwardKLEvaluatorConfig](Evaluator[ConfigType]):
    """Shard by PROBLEM (not trace) so each rank gets complete problems.

    This allows computing per-problem IS metrics locally, then reducing scalars.
    """

    _inference_runner: GPTInferenceRunner
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
        self._inference_runner = GPTInferenceRunner(self._multi_stage, runner=self._runner)
        self._inference_runner.setup()
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
        metrics = self._evaluate()
        safe_barrier(self._distributed.world_group, f"forward_kl_{self._name} end")

        if metrics["num_traces"] == 0:
            return EvaluationMetrics()

        formatted = (
            f"IS Eval ({self._name}): "
            f"acc={metrics['is_accuracy']:.4f}, "
            f"ESS={metrics['mean_ess']:.2f}/{metrics['samples_per_problem']:.1f}, "
            f"({metrics['num_problems']} problems, {metrics['num_traces']} traces)"
        )
        if metrics["num_skipped"] > 0:
            formatted += f" [{metrics['num_skipped']} skipped]"
        log_main_rank(formatted)

        return EvaluationMetrics(
            {f"validation.{self._name}": {k: v for k, v in metrics.items() if k != "num_skipped"}},
            formatted,
        )

    @torch.inference_mode()
    def _evaluate(self) -> dict[str, float]:
        device = self._distributed.device
        data = self._load_traces(device)

        if len(data) == 0:
            return self._reduce_metrics(0.0, 0.0, 0, 0, data.num_skipped)

        batch_size = self._config.batch_size
        student_log_probs_batches: list[torch.Tensor] = []

        for i in range(0, len(data), batch_size):
            batch_log_probs = self._compute_batch_log_probs(
                data.tokens[i : i + batch_size],
                data.prompt_lens[i : i + batch_size],
                data.completion_lens[i : i + batch_size],
            )
            if batch_log_probs is not None:
                student_log_probs_batches.append(batch_log_probs)

        if not student_log_probs_batches:  # non-last PP rank
            return self._reduce_metrics(0.0, 0.0, 0, 0, data.num_skipped)

        student_log_probs = torch.cat(student_log_probs_batches)
        log_w = student_log_probs - data.teacher_log_probs

        log_sum_all = self._scatter_logsumexp(log_w, data.problem_indices, data.num_problems)
        log_w_correct = log_w.masked_fill(~data.corrects, float("-inf"))
        log_sum_correct = self._scatter_logsumexp(log_w_correct, data.problem_indices, data.num_problems)

        # IS accuracy; nan_to_num handles -inf - -inf
        accuracy = (log_sum_correct - log_sum_all).exp().nan_to_num(0.0)

        # ESS = exp(2*logsumexp(log_w) - logsumexp(2*log_w))
        log_sum_sq = self._scatter_logsumexp(2 * log_w, data.problem_indices, data.num_problems)
        ess = (2 * log_sum_all - log_sum_sq).exp().clamp(min=0.0)

        return self._reduce_metrics(
            accuracy.sum().item(),
            ess.sum().item(),
            data.num_problems,
            len(data),
            data.num_skipped,
        )

    def _load_traces(self, device: torch.device) -> TraceTensors:
        import datasets

        ds = datasets.load_dataset(
            self._config.dataset_path,
            split=self._config.split,
            trust_remote_code=self._config.trust_remote_code,
        )

        # Shuffle needed because traces are sorted by problem
        if self._config.num_samples and len(ds) > self._config.num_samples:
            ds = ds.shuffle(seed=self._config.seed).select(range(self._config.num_samples))

        dp_rank = self._distributed.config.data_rank
        dp_size = self._distributed.config.data_parallel

        def belongs_to_shard(example: dict) -> bool:
            h = hashlib.md5(example["problem_id"].encode(), usedforsecurity=False).digest()
            return int.from_bytes(h[:4], "little") % dp_size == dp_rank

        ds = ds.filter(belongs_to_shard)
        traces = list(ds)

        del ds
        gc.collect()

        return TraceTensors.from_traces(traces, self._sequence_length, device)

    def _compute_batch_log_probs(
        self,
        tokens: torch.Tensor,
        prompt_lens: torch.Tensor,
        completion_lens: torch.Tensor,
    ) -> torch.Tensor | None:
        batch_size = tokens.shape[0]
        lm_batch = self._prepare_batch(tokens, prompt_lens, completion_lens)

        with NoAutoValidate():
            batch_config = GPTBatchConfig(
                micro_batch_size=batch_size,
                sequence_length=self._sequence_length,
                micro_sequence_length=self._micro_sequence_length,
                truncate_documents=False,
            )
        batch_config.setup(self._distributed.config)
        batch_config.validate()

        preprocessed_meta = self._multi_stage.base_model.preprocess_meta(batch_config, PhaseType.inference)
        preprocessed = self._multi_stage.base_model.preprocess_batch(
            lm_batch, preprocessed_meta, phase=PhaseType.inference, iteration=0
        )

        # Loop runs through micro-sequences; final kwargs has the logits
        for input_, kwargs in preprocessed:
            kwargs["global_logits"] = True
            self._inference_runner.forward(input_, kwargs)

        if "logits" not in kwargs:  # non-last PP stage
            return None

        logits = kwargs["logits"]
        if kwargs.get(AttentionKwargs.sequence_first, False):
            logits = logits.transpose(0, 1)

        device = logits.device
        seq_len = logits.shape[1]

        pred_logits = logits[:, :-1, :].contiguous()
        targets = tokens[:, 1:].contiguous().to(device)

        # Mask: completion predictions are at [prompt_len-1, prompt_len+completion_len-1)
        mask = self._create_completion_mask(prompt_lens, completion_lens, seq_len - 1)

        ce_loss = F.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view(batch_size, seq_len - 1)

        results = -(ce_loss * mask).sum(dim=1)

        del logits, kwargs, preprocessed, lm_batch

        return results.to(torch.float64)

    def _prepare_batch(
        self,
        tokens: torch.Tensor,
        prompt_lens: torch.Tensor,
        completion_lens: torch.Tensor,
    ) -> LanguageModelBatch:
        samples = []
        for i in range(tokens.shape[0]):
            seq_len = int(prompt_lens[i].item()) + int(completion_lens[i].item())
            sample = LanguageModelSample(TokenSample(tokens[i, :seq_len].cpu()))

            pad_len = self._sequence_length - seq_len
            if pad_len > 0:
                sample = LanguageModelSample.from_documents([sample, sample.get_padding(pad_len)])

            samples.append(sample)

        return LanguageModelBatch.from_samples(samples)

    def _create_completion_mask(
        self,
        prompt_lens: torch.Tensor,
        completion_lens: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        device = prompt_lens.device
        positions = torch.arange(seq_len, device=device)
        start = (prompt_lens - 1).unsqueeze(1)
        end = (prompt_lens + completion_lens - 1).unsqueeze(1)
        return (positions >= start) & (positions < end)

    def _reduce_metrics(
        self, sum_accuracy: float, sum_ess: float, num_problems: int, num_traces: int, num_skipped: int
    ) -> dict[str, float]:
        group = self._distributed.world_group
        sum_accuracy = allreduce_scalar(sum_accuracy, group=group)
        sum_ess = allreduce_scalar(sum_ess, group=group)
        num_problems = int(allreduce_scalar(num_problems, torch.int64, group=group))
        num_traces = int(allreduce_scalar(num_traces, torch.int64, group=group))
        num_skipped = int(allreduce_scalar(num_skipped, torch.int64, group=group))

        if num_problems == 0:
            return {
                "is_accuracy": 0.0,
                "mean_ess": 0.0,
                "samples_per_problem": 0.0,
                "num_traces": 0,
                "num_problems": 0,
                "num_skipped": num_skipped,
            }

        return {
            "is_accuracy": sum_accuracy / num_problems,
            "mean_ess": sum_ess / num_problems,
            "samples_per_problem": num_traces / num_problems,
            "num_traces": num_traces,
            "num_problems": num_problems,
            "num_skipped": num_skipped,
        }

    def _scatter_logsumexp(self, src: torch.Tensor, index: torch.Tensor, num_groups: int) -> torch.Tensor:
        # Max per group for numerical stability
        max_vals = torch.full((num_groups,), float("-inf"), device=src.device, dtype=src.dtype)
        max_vals.scatter_reduce_(0, index, src, reduce="amax")

        src_shifted = (src - max_vals[index]).exp()
        sum_exp = torch.zeros(num_groups, device=src.device, dtype=src.dtype)
        sum_exp.scatter_add_(0, index, src_shifted)

        return max_vals + sum_exp.log()
