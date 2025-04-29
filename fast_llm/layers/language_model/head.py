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
from fast_llm.functional.config import CrossEntropyImpl, TargetFormat, TritonConfig
from fast_llm.functional.cross_entropy import cross_entropy_forward_backward
from fast_llm.functional.linear import output_parallel_linear_backward, output_parallel_linear_forward
from fast_llm.layers.common.auxiliary_loss import AuxiliaryLoss, z_loss
from fast_llm.layers.language_model.config import (
    LanguageModelBaseConfig,
    LanguageModelDimNames,
    LanguageModelKwargs,
    LanguageModelLossNames,
)
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
from fast_llm.logging import log_distributed_tensor
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_
from fast_llm.utils import Assert, div

OUTPUT_WEIGHTS = "output_weights"


class LanguageModelHead[ConfigType: LanguageModelBaseConfig](Configurable[LanguageModelBaseConfig], Layer):
    """
    A language model head (GPT), which combines the final layer norm, logits and cross-entropy (if applicable).
    """

    config_class: typing.ClassVar[type[LanguageModelBaseConfig]] = LanguageModelBaseConfig

    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
        prediction_distance: int,
    ):
        super().__init__(config)
        self._debug_transformer = config.transformer.debug_transformer
        self._tie_word_embeddings = config.tie_word_embeddings
        self._tensor_space = tensor_space

        self._group_size = tensor_space.distributed_config.tensor_parallel
        self._sequence_parallel = tensor_space.distributed_config.sequence_tensor_parallel
        self._parallel_embeddings = tensor_space.distributed_config.tensor_parallel > 1 and config.parallel_embeddings
        self._sequence_parallel_logits = (
            tensor_space.distributed_config.sequence_tensor_parallel and not config.parallel_embeddings
        )
        self._cross_entropy_splits = config.cross_entropy_splits
        if self._cross_entropy_splits is not None and self._sequence_parallel:
            assert not self._parallel_embeddings

        hidden_dim = self._tensor_space.get_tensor_dim(TransformerDimNames.hidden)

        self._loss_name = LanguageModelLossNames.multi_token_prediction_loss(prediction_distance)
        self.final_norm = config.transformer.normalization.get_layer(hidden_dim)
        self._logits_scale_factor = config.logits_scale_factor
        self._z_loss_factor = config.logit_z_loss

        # Distance of the target token prediction
        # 0: next-token prediction
        # >0: multi-token prediction (MTP)
        Assert.geq(prediction_distance, 0)
        self._prediction_distance = prediction_distance
        self._is_last_head = self._prediction_distance == config.prediction_heads - 1

        self._init_output_weights(hidden_dim, config)

        self._cross_entropy_impl = config.cross_entropy_impl
        if self._cross_entropy_impl == CrossEntropyImpl.auto:
            if self._parallel_embeddings:
                self._cross_entropy_impl = CrossEntropyImpl.fused
            elif TritonConfig.TRITON_ENABLED:
                self._cross_entropy_impl = CrossEntropyImpl.triton
            else:
                self._cross_entropy_impl = CrossEntropyImpl.fused

        self._forward = wrap_forward_backward(self._forward_backward, grad_is_context)

        # PEFT.
        self.final_norm = self._config.transformer.peft.apply_other(self.final_norm)
        if hasattr(self, "output_weights"):
            self.output_weights = self._config.transformer.peft.apply_weight(self.output_weights)

    def _init_output_weights(self, hidden_dim: TensorDim, config) -> None:
        # Only the first head defines the output weights
        if self._tie_word_embeddings or self._prediction_distance > 0:
            return
        # untie embedding weights
        vocab_dim = self._tensor_space.get_tensor_dim(
            LanguageModelDimNames.vocab_tp if self._parallel_embeddings else LanguageModelDimNames.vocab
        )
        self.output_weights = ParameterMeta.from_dims(
            (vocab_dim, hidden_dim),
            init_method=init_normal_(
                std=config.init_method_std_embed,
                min_val=config.init_method_min_embed,
                max_val=config.init_method_max_embed,
            ),
        )

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
        if not self._is_last_head:
            # MTP: split the stacked input
            shared_hidden, input_ = torch.unbind(input_, dim=0)
        # TODO: Pytorch copies the grads in backward for no reason (not sure if still the case)
        # TODO: Torch compile implementation sometimes break.
        # TODO: Double-check correctness, optimize a bit more.
        # TODO: Drop autograd entirely.
        # TODO: Skip cross-entropy backward if not needed.
        language_model_loss = self._forward(input_, kwargs, losses)
        if losses is not None and language_model_loss is not None:
            losses[self._loss_name].append(language_model_loss)
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
        target = kwargs.get(
            LanguageModelKwargs.labels
            if self._config.distillation_model is None
            else f"{self._config.distillation_model}_logits"
        )
        if target is not None:
            if self._config.distillation_model is None:
                # MTP: Shift the labels
                target_sequence_length = (
                    target.size(1 - kwargs[TransformerKwargs.sequence_first]) + 1 - self._config.prediction_heads
                )
                if TransformerKwargs.sequence_q_dim in kwargs:
                    Assert.eq(target_sequence_length, kwargs[TransformerKwargs.sequence_q_dim].size)
                target_slice = slice(self._prediction_distance, self._prediction_distance + target_sequence_length)
                target = target[target_slice] if kwargs[TransformerKwargs.sequence_first] else target[:, target_slice]
                target = target.flatten()
            else:
                # Target is reference model logits.
                target = target.flatten(0, -2)

        if self._sequence_parallel_logits:
            target = split_op(target, self._tensor_space.distributed.tensor_group, 0)
        do_grad = target is not None and self.training
        input_ = input_.detach().requires_grad_(do_grad)
        with torch.enable_grad():
            ln_output = self.final_norm(input_)

        grad_output = kwargs[TransformerKwargs.grad_output] / (
            self._group_size if self._sequence_parallel_logits else 1
        )

        output_weights = self._get_output_weights(kwargs)
        loss, ln_output_grad = self._logits_cross_entropy_forward_backward_split(
            ln_output.detach(), target, output_weights, grad_output, kwargs, losses
        )

        if do_grad:
            ln_output.backward(ln_output_grad)
            return loss, input_.grad
        else:
            return loss, None

    def _get_output_weights(self, kwargs: dict) -> torch.Tensor:
        if self._tie_word_embeddings:
            return kwargs[WORD_EMBEDDINGS_WEIGHT]
        if self._prediction_distance > 0:
            return kwargs[OUTPUT_WEIGHTS]
        return self.output_weights

    def _logits_cross_entropy_forward_backward_split(
        self,
        input_: torch.Tensor,
        target: torch.Tensor | None,
        weight: torch.Tensor,
        grad_output: float,
        kwargs: dict,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._cross_entropy_splits is None or target is None:
            loss, logit_input_grad = self._logits_cross_entropy_forward_backward(
                input_, target, weight, grad_output, kwargs, losses
            )
            if target is None:
                # TODO: Make a proper way of returning the model output.
                kwargs["logits" if self._prediction_distance == 0 else f"logits_{self._prediction_distance}"] = loss
                return None, None
        else:
            loss = None
            # TODO MTP: allow a _cross_entropy_splits that is not a divisor of the sequence length
            split_size = div(target.size(0), self._cross_entropy_splits)
            grad_output /= self._cross_entropy_splits
            logit_input = input_.flatten(0, -2)
            logit_input_grad = torch.empty_like(logit_input)
            for logit_input_, target_, logit_input_grad_ in zip(
                logit_input.split(split_size), target.split(split_size), logit_input_grad.split(split_size)
            ):
                loss_, grad_ = self._logits_cross_entropy_forward_backward(
                    logit_input_,
                    target_,
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

    def _logits_cross_entropy_forward_backward(
        self,
        input_: torch.Tensor,
        target: torch.Tensor | None,
        weight: torch.Tensor,
        grad_output: float,
        kwargs: dict,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits, context = output_parallel_linear_forward(
            input_=input_,
            weight=weight,
            bias=None,
            group=self._tensor_space.distributed.tensor_group if self._parallel_embeddings else None,
            sequence_parallel=self._sequence_parallel and self._parallel_embeddings,
        )

        if self._z_loss_factor > 0.0:
            logits = z_loss(
                logits,
                self._z_loss_factor,
                self.training,
                grad_output,
                losses,
                LanguageModelLossNames.z_loss,
                logits_scale_factor=self._logits_scale_factor,
            )
        if self._debug_transformer and self._cross_entropy_splits is None:
            vocab_dim = self._tensor_space.get_tensor_dim(
                LanguageModelDimNames.vocab if self._sequence_parallel_logits else LanguageModelDimNames.vocab_tp
            )
            dims = [*kwargs[TransformerKwargs.hidden_dims][:-1], vocab_dim]
            sequence_index = 1 - int(kwargs[TransformerKwargs.sequence_first])
            dims[sequence_index] = (
                TensorDim(
                    TransformerDimNames.sequence_q_tp, dims[sequence_index].global_size, DistributedDimNames.tensor
                )
                if self._sequence_parallel_logits
                else TensorDim(TransformerDimNames.sequence_q, dims[sequence_index].global_size)
            )

            dim_names = (
                [TransformerDimNames.sequence_q_tp, LanguageModelDimNames.vocab]
                if self._sequence_parallel_logits
                else [TransformerDimNames.sequence_q, LanguageModelDimNames.vocab_tp]
            )

            dim_names.insert(int(kwargs[TransformerKwargs.sequence_first]), TransformerDimNames.batch)
            log_distributed_tensor(
                "",
                logits,
                level=self._debug_transformer,
                meta=TensorMeta.from_dims(tuple(dims), tensor_name="transformer logits", dtype=logits.dtype),
                distributed=self._tensor_space.distributed,
                scale=self._logits_scale_factor,
            )

        if target is None:
            return logits * self._logits_scale_factor, None
        loss, grad = cross_entropy_forward_backward(
            logits.flatten(0, -2),
            target,
            group=self._tensor_space.distributed.tensor_group if self._parallel_embeddings else None,
            grad_output=grad_output,
            implementation=self._cross_entropy_impl,
            logits_scale_factor=self._logits_scale_factor,
            target_format=TargetFormat.labels if self._config.distillation_model is None else TargetFormat.logits,
        )
        # TODO: de-allocate earlier.
        del logits
        return loss, output_parallel_linear_backward(grad, context)
