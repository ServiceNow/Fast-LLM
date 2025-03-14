import typing

import torch
from torch._C._distributed_c10d import ReduceOp  # noqa

from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorSpace
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.layers.common.auxiliary_loss import AuxiliaryLoss
from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.layers.language_model.head import OUTPUT_WEIGHTS, LanguageModelHead
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.tensor import TensorMeta


def grad_is_context_sum(grad_output: torch.Tensor, context: torch.Tensor) -> torch.Tensor:  # noqa
    return grad_output + context


class MultiTokenPredictionTransformerLayer(TransformerLayer):
    """
    A transformer layer for multi-token prediction.
    Takes a shared_hidden as input.
    Returns a stack of shared_hidden and transformer_output.
    """

    # TODO MTP: what layer-index for transformer layer? All the transformer layers in the MTP-LM-heads
    # will have the same name. Is it an issue?
    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> torch.Tensor:
        # TODO MTP: fix dims?
        if isinstance(input_, TensorMeta):
            return self._get_meta(input_, "output", kwargs)
        output = super().forward(input_, kwargs, losses, metrics)
        return torch.stack((input_, output), dim=0)


class MultiTokenPredictionLanguageModelHead(LanguageModelHead):
    """
    A language-model head for multi-token prediction.
    Takes a stacked input (shared_hidden + transformer_layer_output) and returns the shared_hidden.
    """

    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
        multi_token_prediction_index: int,
    ):
        self.multi_token_prediction_index = multi_token_prediction_index
        self.is_last_head = self.multi_token_prediction_index == config.num_multi_token_prediction_heads - 1
        super().__init__(config, tensor_space)
        self.loss_name = LanguageModelLossNames.multi_token_prediction_loss(multi_token_prediction_index)
        # TODO MTP: Handle SP logits and CE splits
        # One issue is that these require the number of labels to be divisible by the nb of splits
        assert not self._sequence_parallel_logits, "Sequence parallel logits not supported for multi-token prediction."
        assert not self._cross_entropy_splits, "Cross-entropy splits not supported for multi-token prediction."

    def _should_init_output_weights(self) -> bool:
        if self.multi_token_prediction_index > 0:
            return False
        return super()._should_init_output_weights()

    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            # TODO MTP: fix dims?
            return TensorMeta.from_tensor_space(
                (DefaultDimNames.scalar,),
                self._tensor_space,
                tensor_name="Loss",
                reductions=((DistributedDimNames.data, ReduceOp.AVG),),  # noqa
            )
        # MTP: split the stacked input
        shared_hidden, transformer_layer_output = torch.unbind(input_, dim=0)
        # TODO: Pytorch copies the grads in backward for no reason (not sure if still the case)
        # TODO: Torch compile implementation sometimes break.
        # TODO: Double-check correctness, optimize a bit more.
        # TODO: Drop autograd entirely.
        # TODO: Skip cross-entropy backward if not needed.
        language_model_loss = self._forward(transformer_layer_output, kwargs, losses)
        if language_model_loss is not None:
            losses[self.loss_name].append(language_model_loss)
        if self.is_last_head:
            # Last layer should return the loss for backward.
            return language_model_loss
        else:
            # Backward hook to compute the gradient of the loss
            shared_hidden = AuxiliaryLoss.apply(shared_hidden, language_model_loss, 1.0)
            # TODO: Return the model output when needed.
            # MTP: Return shared_hidden to be used by the next head.
            return shared_hidden

    def _forward_backward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        labels = kwargs[LanguageModelKwargs.labels] if LanguageModelKwargs.labels in kwargs else None
        # MTP: Shift the labels
        labels = labels[:, self.multi_token_prediction_index :].flatten() if labels is not None else None
        if self._sequence_parallel_logits:
            raise NotImplementedError()
            # TODO MTP
            labels = split_op(labels, self._tensor_space.distributed.tensor_group, 0)
        do_grad = labels is not None and self.training
        input_ = input_.detach().requires_grad_(do_grad)
        with torch.enable_grad():
            # MTP: truncate the input
            if self.multi_token_prediction_index > 0:
                truncated_input = input_[:, : -self.multi_token_prediction_index, :].contiguous()
            else:
                truncated_input = input_
            ln_output = self.final_norm(truncated_input)

        grad_output = kwargs[TransformerKwargs.grad_output] / (
            self._group_size if self._sequence_parallel_logits else 1
        )

        output_weights = self._get_output_weights(kwargs)
        loss, ln_output_grad = self._logits_cross_entropy_forward_backward_split(
            ln_output.detach(), labels, output_weights, grad_output, kwargs, losses
        )

        if do_grad:
            ln_output.backward(ln_output_grad)
            return loss, input_.grad
        else:
            return loss, None

    def _get_output_weights(self, kwargs: dict) -> torch.Tensor:
        if self.multi_token_prediction_index > 0:
            return kwargs[WORD_EMBEDDINGS_WEIGHT] if self._tie_word_embeddings else kwargs[OUTPUT_WEIGHTS]
        return super()._get_output_weights(kwargs)

    # def _logits_cross_entropy_forward_backward_split(
    #     self,
    #     input_: torch.Tensor,
    #     labels: torch.Tensor | None,
    #     weight: torch.Tensor,
    #     grad_output: float,
    #     kwargs: dict,
    #     losses: dict | None = None,
    # ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    #     if self.multi_token_prediction_index > 0:
    #         # This MTP-head is not the first one, so the labels need to be shifted.
    #         if labels is not None:
    #             labels = labels[self.multi_token_prediction_index:]
    #         input_ = input_[:-self.multi_token_prediction_index]
    #     return super()._logits_cross_entropy_forward_backward_split(input_, labels, weight, grad_output, kwargs, losses)
