import abc
import functools
import logging
import typing

import torch
from torch._C._distributed_c10d import ReduceOp  # noqa
from torch.distributed import all_reduce

from fast_llm.core.ops import gather_op, split_op
from fast_llm.engine.base_model.config import LossDef, ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.autograd import grad_is_context, wrap_forward_backward
from fast_llm.functional.linear import output_parallel_linear_backward, output_parallel_linear_forward
from fast_llm.layers.block.block import Block
from fast_llm.layers.block.config import BlockDimNames
from fast_llm.layers.common.auxiliary_loss import AuxiliaryLoss, z_loss
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import (
    LanguageModelEmbeddingsConfig,
    LanguageModelHeadBaseConfig,
    LanguageModelHeadConfig,
)
from fast_llm.layers.language_model.kwargs import LanguageModelKwargs
from fast_llm.layers.language_model.lm_head_losses import _format_name
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert, div, get_unique

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
        if prediction_distance > 0 and (
            self._config.distillation_model is not None or self._config.dpo_reference_model is not None
        ):
            raise NotImplementedError("Multi-token prediction not supported with distillation or dpo.")

        Assert.in_range(prediction_distance, 0, prediction_heads)
        self._prediction_distance = prediction_distance
        self._prediction_heads = prediction_heads
        self._loss_coefficient = loss_coefficient
        self._is_last_head = self._prediction_distance == self._prediction_heads - 1

        self._vocab_parallel = self._distributed_config.tensor_parallel > 1 and embeddings_config.vocab_parallel
        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        self._sequence_parallel_logits = self._sequence_parallel and not self._vocab_parallel
        if self._config.cross_entropy_splits is not None and self._sequence_parallel:
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

        self._formatted_loss_names = {}
        for loss_name, loss_config in self._config.losses.items():
            if loss_config.weight > 0.0:
                self._formatted_loss_names[loss_name] = loss_config.get_formatted_name(
                    loss_name, self._prediction_distance
                )

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

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Add marginal compute? (loss)
        return (
            2
            * (config.forward + 2 * config.backward)
            * (input_.global_shape if config.global_ else input_).numel()
            * (self._vocab_dim.global_size if config.global_ else self._vocab_dim.size)
        )

    def _forward_backward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        targets = self._get_targets(kwargs)
        loss_mask = kwargs.get(LanguageModelKwargs.loss_mask)
        if loss_mask is not None:
            loss_mask = loss_mask.flatten()
            if self._sequence_parallel_logits:
                loss_mask = split_op(loss_mask, self._parallel_dim.group, 0)

        input_ = input_.detach().requires_grad_(do_grad := targets is not None and self.training)
        with torch.enable_grad():
            ln_output = self.final_norm(input_)
            # Transormers expect normalized outputs for the last transformer layer,
            # so we add the norm output to the hidden states.
            self._debug(ln_output, "final_norm", kwargs.get(LanguageModelKwargs.hidden_dims), kwargs)

        grad_output = kwargs[LanguageModelKwargs.grad_output] / (
            self._parallel_dim.size if self._sequence_parallel_logits else 1
        )

        output_weights = self.output_weights
        loss, ln_output_grad = self._logits_cross_entropy_forward_backward_split(
            ln_output.detach(), targets, loss_mask, output_weights, grad_output, kwargs, losses
        )

        if do_grad:
            ln_output.backward(ln_output_grad)
            return loss, input_.grad
        else:
            return loss, None

    def _get_targets(self, kwargs: dict) -> dict | None:
        targets = {}
        for loss_config in self._config.losses.values():
            if loss_config.weight == 0.0:
                continue
            loss_targets = loss_config.extract_targets_from_global_kwargs(
                kwargs,
                prediction_distance=self._prediction_distance,
                prediction_heads=self._prediction_heads,
                head_config=self._config,
                sequence_parallel_logits=self._sequence_parallel_logits,
            )
            targets.update({k: v for k, v in loss_targets.items() if v is not None})
        if len(targets) == 0:
            return None
        return targets

    def get_output_weights(self) -> list[torch.Tensor]:
        return [self.output_weights]

    def _logits_cross_entropy_forward_backward_split(
        self,
        input_: torch.Tensor,
        targets: dict[str, "torch.Tensor"] | None,
        loss_mask: torch.Tensor | None,
        weight: torch.Tensor,
        grad_output: float,
        kwargs: dict,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._config.cross_entropy_splits is None:
            loss, logit_input_grad = self._logits_loss_forward_backward(
                input_, targets, loss_mask, weight, grad_output, kwargs, losses
            )
            if targets is None:
                # TODO: Make a proper way of returning the model output.
                loss = loss.detach()
                if kwargs.get("global_logits"):
                    if self._vocab_parallel:
                        loss = gather_op(loss, self._parallel_dim.group, 2)
                    elif self._sequence_parallel_logits:
                        loss = gather_op(
                            loss, self._parallel_dim.group, 0 if kwargs[LanguageModelKwargs.sequence_first] else 1
                        )
                kwargs["logits" if self._prediction_distance == 0 else f"logits_{self._prediction_distance}"] = loss
                return None, None
        else:
            loss = None
            # TODO MTP: allow a cross_entropy_splits that is not a divisor of the sequence length
            grad_output /= self._config.cross_entropy_splits
            logit_input = input_.flatten(0, -2)
            if self.training:
                logit_input_grad = torch.empty_like(logit_input)
            else:
                logit_input_grad = None

            split_size = div(
                get_unique(target.size(0) for target in targets.values() if target is not None),
                self._config.cross_entropy_splits,
            )
            tensors_split = [
                [None] * self._config.cross_entropy_splits if tensor is None else tensor.split(split_size)
                for tensor in [logit_input, loss_mask, logit_input_grad]
            ]
            target_split = {
                name: (
                    [None] * self._config.cross_entropy_splits
                    if targets[name] is None
                    else targets[name].split(split_size)
                )
                for name in targets
            }

            for i, (logit_input_, loss_mask_, logit_input_grad_) in enumerate(zip(*tensors_split, strict=True)):
                loss_, grad_ = self._logits_loss_forward_backward(
                    logit_input_,
                    {name: target_split[name][i] for name in target_split},
                    loss_mask_,
                    weight,
                    grad_output,
                    kwargs,
                )
                # TODO: Avoid copy with explicit out argument.
                if self.training:
                    logit_input_grad_.copy_(grad_)
                loss = loss_ if loss is None else loss + loss_
                del grad_, loss_
        loss_count = (self._config.cross_entropy_splits or 1) * (
            self._parallel_dim.size if self._sequence_parallel_logits else 1
        )
        if loss_count != 1:
            loss.div_(loss_count)
        if self._sequence_parallel_logits:
            # TODO: Async
            all_reduce(loss, group=self._parallel_dim.group)
        return loss, logit_input_grad.view_as(input_) if logit_input_grad is not None else None

    def _logits_loss_forward_backward(
        self,
        input_: torch.Tensor,
        targets: dict[str, "torch.Tensor"] | None,
        loss_mask: torch.Tensor | None,
        weight: torch.Tensor,
        grad_output: float,
        kwargs: dict,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        group = self._parallel_dim.group if self._vocab_parallel else None
        logits, context = output_parallel_linear_forward(
            input_=input_,
            weight=weight,
            bias=None,
            group=group,
            sequence_parallel=self._sequence_parallel and self._vocab_parallel,
        )

        # TODO: also move to lm_head_losses?
        if self._config.logit_z_loss > 0.0:
            logits = z_loss(
                logits,
                self._config.logit_z_loss,
                self.training,
                grad_output,
                losses,
                self._z_loss_name,
                logits_scale_factor=self._config.logits_scale_factor,
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

        if targets is None:
            return logits * self._config.logits_scale_factor, None

        total_loss, grad = None, None
        for loss_name, loss_config in self._config.losses.items():
            if loss_config.weight == 0.0:
                continue
            # losses are returned unscaled but the grads are already scaled
            loss_unscaled_, grad_ = loss_config.compute_loss(
                logits,
                loss_mask,
                grad_output=(
                    (grad_output * self._loss_coefficient * loss_config.weight if grad_output is not None else None)
                    if loss_config.weight != 0.0
                    else None
                ),
                group=group,
                logits_scale_factor=self._config.logits_scale_factor,
                vocab_parallel=self._vocab_parallel,
                kwargs={**kwargs, **targets},
            )
            loss_ = loss_unscaled_ * loss_config.weight * self._loss_coefficient

            if losses is not None:
                losses[self._formatted_loss_names[loss_name]].append(loss_unscaled_.detach())

            if total_loss is None:
                total_loss = loss_
            else:
                total_loss = total_loss + loss_

            if grad_ is not None:
                if grad is None:
                    grad = grad_
                else:
                    grad = grad + grad_

        if losses is not None and total_loss is not None:
            losses[self._total_head_loss_name].append(total_loss.detach())

        return total_loss, output_parallel_linear_backward(grad, context) if self.training else None

    @functools.cached_property
    def _total_head_loss_name(self) -> str:
        """
        Combined total scaled loss used for training.
        """
        name = "lm_head_loss"
        if self._prediction_distance > 0:
            name = f"{name}_{self._prediction_distance}"
        return name

    @functools.cached_property
    def _z_loss_name(self) -> str:
        name = "z_loss"
        if self._prediction_distance > 0:
            name = f"{name}_{self._prediction_distance}"
        return name

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        loss_defs = [
            LossDef(
                name=self._total_head_loss_name, formatted_name=_format_name(self._total_head_loss_name), count=count
            )
        ]
        if self._config.logit_z_loss > 0.0:
            loss_defs.append(
                LossDef(name=self._z_loss_name, formatted_name=_format_name(self._z_loss_name), count=count)
            )
        for loss_name, loss_config in self._config.losses.items():
            loss_def: LossDef = loss_config.get_loss_def(
                name=loss_name, count=count, prediction_distance=self._prediction_distance
            )
            loss_defs.append(loss_def)

        return loss_defs

    @property
    def heads(self):
        # For compatibility with MTP.
        return [self]
