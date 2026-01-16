import abc
import logging
import typing

import torch
from torch._C._distributed_c10d import ReduceOp  # noqa
from torch.distributed import all_reduce

from fast_llm.core.ops import gather_op, split_op
from fast_llm.engine.base_model.config import LossDef, ResourceUsageConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.autograd import grad_is_context, wrap_forward_backward
from fast_llm.functional.linear import output_parallel_linear_backward, output_parallel_linear_forward
from fast_llm.layers.block.block import Block
from fast_llm.layers.block.config import BlockDimNames
from fast_llm.layers.common.auxiliary_loss import AuxiliaryLoss
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import (
    LM_HEAD_LOSS_NAME,
    LanguageModelEmbeddingsConfig,
    LanguageModelHeadBaseConfig,
    LanguageModelHeadConfig,
    LanguageModelKwargs,
)
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)

OUTPUT_WEIGHTS = "output_weights"


class LanguageModelHeadBase[ConfigType: LanguageModelHeadBaseConfig](Block[ConfigType]):
    @abc.abstractmethod
    def get_output_weights(self) -> list[torch.Tensor]:
        pass


class LanguageModelHead[ConfigType: LanguageModelHeadConfig](LanguageModelHeadBase[ConfigType]):
    """
    A language model head (GPT), which combines the final layer norm, logits and cross-entropy (if applicable).
    TODO: Cleanup (dynamic type? composition?)
    """

    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        embeddings_config: LanguageModelEmbeddingsConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        prediction_distance: int = 0,
        prediction_heads: int = 1,
        loss_coefficient: float = 1.0,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )
        Assert.in_range(prediction_distance, 0, prediction_heads)
        self._prediction_distance = prediction_distance
        self._prediction_heads = prediction_heads
        self._loss_coefficient = loss_coefficient
        self._is_last_head = self._prediction_distance == self._prediction_heads - 1

        self._vocab_parallel = self._distributed_config.tensor_parallel > 1 and embeddings_config.vocab_parallel
        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        self._sequence_parallel_logits = self._sequence_parallel and not self._vocab_parallel
        if self._config.cross_entropy_splits > 1 and self._sequence_parallel:
            assert not self._vocab_parallel

        self._forward = wrap_forward_backward(self._forward_backward, grad_is_context)

        self.final_norm = self._config.normalization.get_layer(
            self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft
        )

        self._vocab_dim = TensorDim(
            "vocab", embeddings_config.vocab_size, self._parallel_dim if self._vocab_parallel else None
        )
        self.output_weights = self._config.output_weight.get_parameter(
            (self._vocab_dim, self._hidden_dim),
            default_initialization=init_normal_(std=self._hidden_size**-0.5),
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Add marginal compute? (loss)
        return (
            2
            * (config.forward + 2 * config.backward)
            * (input_.global_shape if config.global_ else input_).numel()
            * (self._vocab_dim.global_size if config.global_ else self._vocab_dim.size)
        )

    def get_output_weights(self) -> list[torch.Tensor]:
        return [self.output_weights]

    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            if self._is_last_head:
                return TensorMeta.from_dims(
                    (scalar_dim,),
                    tensor_name=f"{self.module_name} output",
                    reductions=(
                        (self._distributed_config.get_distributed_dim(DistributedDimNames.data), ReduceOp.AVG),
                    ),
                )
            else:
                return TensorMeta.from_dims(input_.dims[1:], tensor_name="Shared hidden")

        if not self._is_last_head:
            # MTP: split the stacked input
            shared_hidden, input_ = torch.unbind(input_, dim=0)
        # TODO: Pytorch copies the grads in backward for no reason (not sure if still the case)
        # TODO: Torch compile implementation sometimes break.
        # TODO: Double-check correctness, optimize a bit more.
        # TODO: Drop autograd entirely.
        # TODO: Skip cross-entropy backward if not needed.
        language_model_loss = self._forward(input_, kwargs, losses)
        # TODO: Return the model output when needed.
        if self._is_last_head:
            # Last head should return the loss for backward.
            return language_model_loss
        else:
            if self.training:
                # Backward hook to compute the gradient of the loss
                shared_hidden = AuxiliaryLoss.apply(shared_hidden, language_model_loss, 1.0)
            # MTP: Return shared_hidden to be used by the next head.
            return shared_hidden

    def _forward_backward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_ = input_.detach().requires_grad_(self.training)
        with torch.enable_grad():
            ln_output = self.final_norm(input_)
            # Transformers expect normalized outputs for the last transformer layer,
            # so we add the norm output to the hidden states.
            self._debug(ln_output, "final_norm", kwargs.get(LanguageModelKwargs.hidden_dims), kwargs)
        loss, ln_output_grad = self._logits_loss_forward_backward(ln_output.detach().flatten(0, -2), kwargs, losses)
        if ln_output_grad is None:
            return loss, None
        else:
            ln_output.backward(ln_output_grad.view_as(ln_output))
            return loss, input_.grad

    def _logits_loss_forward_backward(
        self,
        input_: torch.Tensor,
        kwargs: dict,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        loss_mask = kwargs.get(LanguageModelKwargs.loss_mask)
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
            if self._sequence_parallel_logits:
                loss_mask = split_op(loss_mask, self._parallel_dim.group, 0)

        if not self.training:
            logits, _ = self._logits_loss_forward_backward_partial(input_, loss_mask, kwargs, return_logits=True)
            # TODO: Make a proper way of returning the model output.
            logits = logits.detach()
            if kwargs.get("global_logits"):
                if self._vocab_parallel:
                    logits = gather_op(logits, self._parallel_dim.group, 2)
                elif self._sequence_parallel_logits:
                    logits = gather_op(
                        logits, self._parallel_dim.group, 0 if kwargs[LanguageModelKwargs.sequence_first] else 1
                    )
            kwargs["logits" if self._prediction_distance == 0 else f"logits_{self._prediction_distance}"] = (
                logits.detach()
            )
            return None, None
        elif self._config.cross_entropy_splits == 1:
            losses_, input_grad = self._logits_loss_forward_backward_partial(input_, loss_mask, kwargs)
        else:
            input_grad = torch.empty_like(input_)
            tensors_split = [
                (
                    [None] * self._config.cross_entropy_splits
                    if tensor is None
                    else tensor.chunk(self._config.cross_entropy_splits)
                )
                for tensor in [input_, loss_mask, input_grad]
            ]
            for split_index, (partial_input_, loss_mask_, input_grad_) in enumerate(zip(*tensors_split, strict=True)):
                partial_losses_, grad_ = self._logits_loss_forward_backward_partial(
                    partial_input_,
                    loss_mask_,
                    kwargs,
                    split_index=split_index,
                )
                # TODO: Avoid copy with explicit out argument.
                input_grad_.copy_(grad_)
                if split_index == 0:
                    losses_ = partial_losses_
                else:
                    for name in self._config.losses:
                        losses_[name] += partial_losses_[name]

        loss: torch.Tensor = sum(
            (self.config.losses[name].weight * self._loss_coefficient / self._config.cross_entropy_splits) * loss_
            for name, loss_ in losses_.items()
        )
        if self._sequence_parallel_logits:
            # TODO: Async
            all_reduce(loss, op=ReduceOp.AVG, group=self._parallel_dim.group)

        if losses is not None:
            losses[self.get_full_loss_name(LM_HEAD_LOSS_NAME)].append(loss)
            if len(self._config.losses) > 1:
                for name, loss_ in losses_.items():
                    if self._config.cross_entropy_splits != 1:
                        loss_ /= self._config.cross_entropy_splits
                    if self._sequence_parallel_logits:
                        # TODO: Async
                        all_reduce(loss_, op=ReduceOp.AVG, group=self._parallel_dim.group)
                    losses[name].append(loss_)

        return loss, input_grad

    def _logits_loss_forward_backward_partial(
        self,
        input_: torch.Tensor,
        loss_mask: torch.Tensor | None,
        kwargs: dict,
        split_index: int = 0,
        return_logits: bool = False,
    ) -> tuple[dict[str, torch.Tensor] | torch.Tensor, torch.Tensor | None]:
        group = self._parallel_dim.group if self._vocab_parallel else None
        logits, context = output_parallel_linear_forward(
            input_=input_,
            weight=self.output_weights,
            bias=None,
            group=group,
            sequence_parallel=self._sequence_parallel and self._vocab_parallel,
        )

        sequence_dim = BlockDimNames.sequence_q_tp if self._sequence_parallel_logits else BlockDimNames.sequence_q
        if LanguageModelKwargs.hidden_dims in kwargs:
            batch_dim = kwargs[LanguageModelKwargs.hidden_dims][1 if kwargs[LanguageModelKwargs.sequence_first] else 0]
            dims = (
                (sequence_dim, batch_dim, self._vocab_dim)
                if kwargs[LanguageModelKwargs.sequence_first]
                else (batch_dim, sequence_dim, self._vocab_dim)
            )
        else:
            dims = None
        self._debug(logits, "logits", dims, kwargs, scale=self._config.logits_scale_factor)

        if return_logits:
            return logits, None

        losses, grad = {}, None
        for loss_name, loss_config in self._config.losses.items():
            # losses are returned unscaled but the grads are already scaled
            # TODO: ====== grad_output can't be None?
            grad_output = kwargs.get(LanguageModelKwargs.grad_output)
            if grad_output is not None:
                grad_output = (
                    grad_output
                    * self._loss_coefficient
                    * loss_config.weight
                    / (self._parallel_dim.size if self._sequence_parallel_logits else 1)
                    / self._config.cross_entropy_splits
                )
            loss, grad_ = loss_config.get_loss(
                logits,
                loss_mask,
                grad_output=None if grad_output == 0.0 else grad_output,
                group=group,
                logits_scale_factor=self._config.logits_scale_factor,
                kwargs=kwargs,
                prediction_distance=self._prediction_distance,
                prediction_heads=self._prediction_heads,
                split_index=split_index,
                num_splits=self._config.cross_entropy_splits,
                sequence_parallel_logits=self._sequence_parallel_logits,
            )
            losses[loss_name] = loss.detach()
            if grad_ is not None:
                # TODO: Accumulate grads in-place to reduce memory and compute overhead.
                grad = grad_ if grad is None else grad + grad_

        return losses, output_parallel_linear_backward(grad, context) if self.training else None

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        return [
            LossDef(name=(name := self.get_full_loss_name(LM_HEAD_LOSS_NAME)), formatted_name=name, count=count),
            *(
                LossDef(
                    name=(name_ := self.get_full_loss_name(name)),
                    formatted_name=name_,
                    count=count,
                    dtype=DataType.float32,
                )
                for name, loss_config in self._config.losses.values()
            ),
        ]

    def get_full_loss_name(self, name) -> str:
        if self._prediction_distance > 0:
            name = f"{name}_{self._prediction_distance}"
        return name

    @property
    def heads(self):
        # For compatibility with MTP.
        return [self]
