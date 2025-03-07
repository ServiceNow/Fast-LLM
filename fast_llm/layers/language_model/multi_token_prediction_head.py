import typing

import torch
from torch._C._distributed_c10d import ReduceOp  # noqa
from torch.distributed import all_reduce

from fast_llm.config import Configurable
from fast_llm.core.ops import split_op
from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.functional.autograd import grad_is_context, wrap_forward_backward
from fast_llm.functional.config import CrossEntropyImpl, TritonConfig
from fast_llm.functional.cross_entropy import cross_entropy_forward_backward
from fast_llm.functional.linear import output_parallel_linear_backward, output_parallel_linear_forward
from fast_llm.layers.common.auxiliary_loss import z_loss
from fast_llm.layers.language_model.config import (
    LanguageModelBaseConfig,
    LanguageModelDimNames,
    LanguageModelKwargs,
    LanguageModelLossNames,
)
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.layers.language_model.head import OUTPUT_WEIGHTS, LanguageModelHead
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.logging import log_distributed_tensor
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_
from fast_llm.utils import div

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
        super().__init__(config, tensor_space)
        self.multi_token_prediction_index = multi_token_prediction_index
        self.loss_name = LanguageModelLossNames.multi_token_prediction_loss(multi_token_prediction_index)

        # TODO MTP: Handle tied output weights
    
    def _should_init_output_weights(self) -> bool:
        if self.multi_token_prediction_index > 0:
            return False
        return super()._should_init_output_weights()
    
    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
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
        # TODO: Return the model output when needed.
        # MTP: Return shared_hidden to be used by the next head.
        return shared_hidden
    
    def _get_output_weights(self, kwargs: dict) -> torch.Tensor:
        if self.multi_token_prediction_index > 0:
            return kwargs[WORD_EMBEDDINGS_WEIGHT] if self._tied_output_weights else kwargs[OUTPUT_WEIGHTS]
        return super()._get_output_weights(kwargs)
    
    def _logits_cross_entropy_forward_backward_split(
        self,
        input_: torch.Tensor,
        labels: torch.Tensor | None,
        weight: torch.Tensor,
        grad_output: float,
        kwargs: dict,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._cross_entropy_splits is None or labels is None:
            loss, logit_input_grad = self._logits_cross_entropy_forward_backward(
                input_, labels, weight, grad_output, kwargs, losses
            )
            if labels is None:
                # TODO: Make a proper way of returning the model output.
                kwargs["logits"] = loss
                return None, None
        else:
            loss = None
            split_size = div(labels.numel(), self._cross_entropy_splits)
            grad_output /= self._cross_entropy_splits
            logit_input = input_.flatten(0, -2)
            logit_input_grad = torch.empty_like(logit_input)
            # TODO MTP: label splitting
            for logit_input_, labels_, logit_input_grad_ in zip(
                logit_input.split(split_size), labels.split(split_size), logit_input_grad.split(split_size)
            ):
                loss_, grad_ = self._logits_cross_entropy_forward_backward(
                    logit_input_,
                    labels_,
                    weight,
                    grad_output,
                    kwargs,
                )
                # TODO: Avoid copy with explicit out argument.
                logit_input_grad_.copy_(grad_)
                loss = loss_ if loss is None else loss + loss_
                del grad_, loss_
        loss_count = (self._cross_entropy_splits or 1) * (self._group_size if self._sequence_parallel_logits else 1)
        if loss_count != 1:
            loss.div_(loss_count)
        if self._sequence_parallel_logits:
            # TODO: Async
            all_reduce(loss, group=self._tensor_space.distributed.tensor_group)
        return loss, logit_input_grad.view_as(input_)
