import typing

import torch

from fast_llm.functional.config import TargetFormat, TritonConfig
from fast_llm.functional.entropy_loss import fused_entropy_loss_forward_backward
from fast_llm.functional.triton.entropy_loss import triton_entropy_loss_forward_backward
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.loss.config import (
    LanguageModelDistillationLossConfig,
    LanguageModelLabelEntropyLossConfig,
)
from fast_llm.layers.language_model.loss.loss import LanguageModelLoss
from fast_llm.utils import Assert


class LanguageModelLabelEntropyLoss[ConfigType: LanguageModelLabelEntropyLossConfig](LanguageModelLoss[ConfigType]):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        return (
            triton_entropy_loss_forward_backward
            if TritonConfig.enabled(logits.device, self._config.use_triton)
            else fused_entropy_loss_forward_backward
        )(
            logits,
            self._get_labels(kwargs, split_index),
            None,  # Labels are already masked
            grad_logits=grad_logits,
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.labels,
            entropy_loss_type=self._config.loss_type,
            divisor=self._get_label_count(kwargs),
        )


class LanguageModelDistillationLoss[ConfigType: LanguageModelDistillationLossConfig](LanguageModelLoss[ConfigType]):
    def _forward_backward(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        losses: dict | None = None,
        split_index: int = 0,
        grad_logits: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        # Parallel teacher stream: student and teacher have different sequence
        # lengths, so we gather both to a flat [N_total, V] tensor at masked
        # positions before running KL.
        if kwargs.get(LanguageModelKwargs.teacher_loss_mask) is not None:
            return self._forward_backward_parallel_stream(logits, kwargs, split_index, grad_logits)

        return (
            triton_entropy_loss_forward_backward
            if TritonConfig.enabled(logits.device, self._config.use_triton)
            else fused_entropy_loss_forward_backward
        )(
            logits,
            self._get_reference_model_logits(self._config.reference_model, kwargs, split_index),
            self._get_loss_mask(kwargs, split_index),
            grad_output=self._get_grad_output(kwargs),
            grad_logits=grad_logits,
            group=self._parallel_dim.group if self._vocab_parallel else None,
            logits_scale_factor=self._logits_scale_factor,
            temperature=self._config.temperature,
            target_format=TargetFormat.logits,
            entropy_loss_type=self._config.loss_type,
            divisor=self._get_label_count(kwargs),
        )

    def _forward_backward_parallel_stream(
        self,
        logits: "torch.Tensor",
        kwargs: dict[str, typing.Any],
        split_index: int,
        grad_logits: torch.Tensor | None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        # SP at the loss layer would require an uneven all-gather of the masked
        # tokens across the TP group, since student and teacher shard boundaries
        # no longer correspond. Not implemented yet.
        assert not self._sequence_parallel, (
            "LanguageModelDistillationLoss with parallel teacher stream is not yet supported "
            "under sequence-tensor-parallel logits. Set embeddings.vocab_parallel=true."
        )

        student_mask_chunk = self._get_loss_mask(kwargs, split_index)
        student_flat = logits[student_mask_chunk]                              # [N_chunk, V]
        teacher_flat = self._get_teacher_flat_chunk(kwargs, split_index, student_flat.shape[0])
        Assert.eq(teacher_flat.shape, student_flat.shape)

        loss, grad_flat = (
            triton_entropy_loss_forward_backward
            if TritonConfig.enabled(logits.device, self._config.use_triton)
            else fused_entropy_loss_forward_backward
        )(
            student_flat,
            teacher_flat,
            None,
            grad_logits=None,
            grad_output=self._get_grad_output(kwargs),
            group=self._parallel_dim.group if self._vocab_parallel else None,
            logits_scale_factor=self._logits_scale_factor,
            temperature=self._config.temperature,
            target_format=TargetFormat.logits,
            entropy_loss_type=self._config.loss_type,
            divisor=self._get_label_count(kwargs),
        )

        del student_flat, teacher_flat

        if grad_flat is not None:
            if grad_logits is None:
                grad_logits = torch.zeros_like(logits)

            mask_indices = student_mask_chunk.nonzero().squeeze(-1)
            grad_logits.index_add_(0, mask_indices, grad_flat)

        return loss, grad_logits

    def _get_teacher_flat_chunk(
        self, kwargs: dict[str, typing.Any], split_index: int, n_chunk: int
    ) -> torch.Tensor:
        """Return the [n_chunk, V] slice of the global teacher flat logits aligned with this chunk."""
        ref_model = self._config.reference_model
        logits_name = self.module_name.rsplit(".", 2)[0] + ".logits"
        flat_key = f"_distill_teacher_flat::{ref_model}::{logits_name}"
        if flat_key not in kwargs:
            Assert.incl(logits_name, kwargs[f"reference_{ref_model}_hidden_states"])
            teacher_logits = kwargs[f"reference_{ref_model}_hidden_states"][logits_name]
            teacher_mask = kwargs[LanguageModelKwargs.teacher_loss_mask][self._prediction_distance - 1]
            kwargs[flat_key] = teacher_logits[teacher_mask]

        offsets_key = f"_distill_chunk_offsets::{self._prediction_distance}"
        if offsets_key not in kwargs:
            student_mask_full = kwargs[LanguageModelKwargs.loss_mask][self._prediction_distance - 1]
            chunk_masks = (
                student_mask_full.chunk(self._num_splits) if self._num_splits > 1 else (student_mask_full,)
            )
            offsets = [0]
            for chunk_mask in chunk_masks:
                offsets.append(offsets[-1] + int(chunk_mask.sum().item()))
            kwargs[offsets_key] = offsets

        start = kwargs[offsets_key][split_index]
        return kwargs[flat_key][start : start + n_chunk]

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return {"return_prediction_mask": True}
