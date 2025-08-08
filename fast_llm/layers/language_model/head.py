import logging
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
from fast_llm.functional.config import CrossEntropyImpl, DistillationLossImpl, TargetFormat, TritonConfig
from fast_llm.functional.cross_entropy import cross_entropy_forward_backward, reverse_kl_forward_backward
from fast_llm.functional.dpo import compute_dpo_loss
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
from fast_llm.utils import Assert, div, get_unique

logger = logging.getLogger(__name__)

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

        hidden_dim = self._tensor_space[TransformerDimNames.hidden]

        self._loss_coefficient = (
            config.prediction_loss_coefficient[prediction_distance] if config.prediction_loss_coefficient else 1.0
        )
        self._loss_name = LanguageModelLossNames.multi_token_prediction_loss(prediction_distance)
        self.final_norm = config.transformer.normalization.get_layer(hidden_dim)
        self._logits_scale_factor = config.logits_scale_factor
        self._language_model_loss_factor = config.language_model_loss_factor
        self._distillation_loss_factor = config.distillation_loss_factor
        self._z_loss_factor = config.logit_z_loss

        # Distance of the target token prediction
        # 0: next-token prediction
        # >0: multi-token prediction (MTP)
        Assert.geq(prediction_distance, 0)
        self._prediction_distance = prediction_distance
        self._is_last_head = self._prediction_distance == config.prediction_heads - 1

        self._init_output_weights(hidden_dim, config)

        self._use_dpo_loss = config.enable_dpo
        if self._use_dpo_loss:
            self.dpo_beta = config.dpo_beta
        else:
            self._cross_entropy_impl = config.cross_entropy_impl
            self._distillation_loss_implementation = config.distillation_loss_implementation
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
        vocab_dim = self._tensor_space[
            LanguageModelDimNames.vocab_tp if self._parallel_embeddings else LanguageModelDimNames.vocab
        ]
        self.output_weights = ParameterMeta.from_dims(
            (vocab_dim, hidden_dim),
            init_method=init_normal_(
                std=config.init_method_std_embed,
                min_val=config.init_method_min_embed,
                max_val=config.init_method_max_embed,
            ),
            lr_scale=config.output_lr_scale,
        )

    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            if self._is_last_head:
                return TensorMeta.from_tensor_space(
                    (DefaultDimNames.scalar,),
                    self._tensor_space,
                    tensor_name="Loss",
                    reductions=((DistributedDimNames.data, ReduceOp.AVG),),  # noqa
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
        if losses is not None and language_model_loss is not None:
            losses[self._loss_name].append(language_model_loss.detach())
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
        targets = self._get_targets(kwargs)
        input_ = input_.detach().requires_grad_(do_grad := targets is not None and self.training)
        with torch.enable_grad():
            ln_output = self.final_norm(input_)

            if "output_hidden_states" in kwargs and kwargs["output_hidden_states"]:
                # The last hidden layer output is returned normalized in the HF Transformers-style output, at least for LLama style models.
                # So, if needed, we gather the data after normalization and set it as the output of the previous layer.
                dims = list(kwargs[TransformerKwargs.hidden_dims])
                sequence_index = 1 - int(kwargs[TransformerKwargs.sequence_first])
                dims[sequence_index] = (
                    TensorDim(
                        TransformerDimNames.sequence_q_tp, dims[sequence_index].global_size, DistributedDimNames.tensor
                    )
                    if self._sequence_parallel_logits
                    else TensorDim(TransformerDimNames.sequence_q, dims[sequence_index].global_size)
                )
                meta = TensorMeta.from_dims(tuple(dims), tensor_name="transformer hidden_state", dtype=ln_output.dtype)
                hidden_state, _ = meta.local_to_global(ln_output.detach(), distributed=self._tensor_space.distributed)
                kwargs["hidden_states"][len(kwargs["hidden_states"]) - 1]["tensor"] = hidden_state

        grad_output = kwargs[TransformerKwargs.grad_output] / (
            self._group_size if self._sequence_parallel_logits else 1
        )

        output_weights = self._get_output_weights(kwargs)
        loss, ln_output_grad = self._logits_cross_entropy_forward_backward_split(
            ln_output.detach(), targets, output_weights, grad_output, kwargs, losses
        )

        if do_grad:
            ln_output.backward(ln_output_grad)
            return loss, input_.grad
        else:
            return loss, None

    def _get_targets(
        self, kwargs: dict
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None:
        # Loss mask for distillation. (Labels are already masked.)
        if self._use_dpo_loss:
            dpo_target = kwargs.get(LanguageModelKwargs.labels)
            lm_target = None
            distillation_target = None
            loss_mask = None
        else:
            dpo_target = None
            if self._config.distillation_model is None:
                distillation_target, loss_mask = None, None
            else:
                # Target is reference model logits.
                distillation_target = kwargs[f"{self._config.distillation_model}_logits"].flatten(0, -2)
                loss_mask = kwargs.get(LanguageModelKwargs.loss_mask)
                if loss_mask is not None:
                    loss_mask = loss_mask.flatten()

            if self._config.distillation_model is None or self._language_model_loss_factor > 0.0:
                lm_target = kwargs.get(LanguageModelKwargs.labels)
                if lm_target is not None:
                    # MTP: Shift the labels
                    lm_target_sequence_length = (
                        lm_target.size(1 - kwargs[TransformerKwargs.sequence_first])
                        + 1
                        - self._config.prediction_heads
                    )
                    if TransformerKwargs.sequence_q_dim in kwargs:
                        Assert.eq(lm_target_sequence_length, kwargs[TransformerKwargs.sequence_q_dim].size)
                    lm_target_slice = slice(
                        self._prediction_distance, self._prediction_distance + lm_target_sequence_length
                    )
                    lm_target = (
                        lm_target[lm_target_slice]
                        if kwargs[TransformerKwargs.sequence_first]
                        else lm_target[:, lm_target_slice]
                    ).flatten()
            else:
                lm_target = None

        targets = (dpo_target, lm_target, distillation_target, loss_mask)
        if self._sequence_parallel_logits:
            targets = [
                None if target is None else split_op(target, self._tensor_space.distributed.tensor_group, 0)
                for target in targets
            ]
        if not any(target is not None for target in targets):
            # Simplify so we don't have to check every time.
            targets = None
        return targets

    def _get_output_weights(self, kwargs: dict) -> torch.Tensor:
        if self._tie_word_embeddings:
            return kwargs[WORD_EMBEDDINGS_WEIGHT]
        if self._prediction_distance > 0:
            return kwargs[OUTPUT_WEIGHTS]
        return self.output_weights

    def _logits_cross_entropy_forward_backward_split(
        self,
        input_: torch.Tensor,
        targets: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None,
        weight: torch.Tensor,
        grad_output: float,
        kwargs: dict,
        losses: dict | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._cross_entropy_splits is None or targets is None:
            loss, logit_input_grad = self._logits_cross_entropy_forward_backward(
                input_, targets, weight, grad_output, kwargs, losses
            )
            if targets is None:
                # TODO: Make a proper way of returning the model output.
                kwargs["logits" if self._prediction_distance == 0 else f"logits_{self._prediction_distance}"] = loss
                return None, None
        else:
            loss = None
            # TODO MTP: allow a _cross_entropy_splits that is not a divisor of the sequence length
            grad_output /= self._cross_entropy_splits
            logit_input = input_.flatten(0, -2)
            if self.training:
                logit_input_grad = torch.empty_like(logit_input)
            else:
                logit_input_grad = None
            split_size = div(
                get_unique(target.size(0) for target in targets if target is not None), self._cross_entropy_splits
            )
            tensors_split = [
                [None] * self._cross_entropy_splits if tensor is None else tensor.split(split_size)
                for tensor in [logit_input, *targets, logit_input_grad]
            ]
            for logit_input_, *targets_, logit_input_grad_ in zip(*tensors_split, strict=True):
                loss_, grad_ = self._logits_cross_entropy_forward_backward(
                    logit_input_,
                    targets_,
                    weight,
                    grad_output,
                    kwargs,
                )
                # TODO: Avoid copy with explicit out argument.
                if self.training:
                    logit_input_grad_.copy_(grad_)
                loss = loss_ if loss is None else loss + loss_
                del grad_, loss_
        loss_count = (self._cross_entropy_splits or 1) * (self._group_size if self._sequence_parallel_logits else 1)
        if loss_count != 1:
            loss.div_(loss_count)
        if self._sequence_parallel_logits:
            # TODO: Async
            all_reduce(loss, group=self._tensor_space.distributed.tensor_group)
        return loss, logit_input_grad.view_as(input_) if logit_input_grad is not None else None

    def _logits_cross_entropy_forward_backward(
        self,
        input_: torch.Tensor,
        targets: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
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
            vocab_dim = self._tensor_space[
                LanguageModelDimNames.vocab if self._sequence_parallel_logits else LanguageModelDimNames.vocab_tp
            ]
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

        if targets is None:
            return logits * self._logits_scale_factor, None
        dpo_target, lm_target, distillation_target, loss_mask = targets

        if dpo_target is not None:
            dpo_loss, dpo_grad = compute_dpo_loss(
                logits,
                dpo_target,
                kwargs.get(f"{self._config.dpo_reference_model}_logits"),
                kwargs[LanguageModelKwargs.chosen_spans],
                kwargs[LanguageModelKwargs.rejected_spans],
                self.dpo_beta,
                grad_output * self._loss_coefficient,
            )
        else:
            dpo_loss, dpo_grad = None, None

        if lm_target is not None:
            lm_loss, lm_grad = cross_entropy_forward_backward(
                logits.flatten(0, -2),
                lm_target,
                None,
                group=self._tensor_space.distributed.tensor_group if self._parallel_embeddings else None,
                grad_output=grad_output * self._loss_coefficient * self._language_model_loss_factor,
                implementation=self._cross_entropy_impl,
                logits_scale_factor=self._logits_scale_factor,
                target_format=TargetFormat.labels,
            )
            lm_loss = lm_loss * self._language_model_loss_factor
        else:
            lm_loss, lm_grad = None, None

        if distillation_target is not None and self._distillation_loss_factor > 0.0:
            if self._distillation_loss_implementation == DistillationLossImpl.reverse_kl:
                distillation_loss, distillation_grad = reverse_kl_forward_backward(
                    logits.flatten(0, -2),
                    distillation_target,
                    loss_mask,
                    grad_output=grad_output * self._loss_coefficient * self._distillation_loss_factor,
                    group=self._tensor_space.distributed.tensor_group if self._parallel_embeddings else None,
                    logits_scale_factor=self._logits_scale_factor,
                    teacher_softmax_temperature=self._config.teacher_softmax_temperature,
                    target_format=(
                        TargetFormat.labels if self._config.distillation_model is None else TargetFormat.logits
                    ),
                )
            elif self._distillation_loss_implementation == DistillationLossImpl.cross_entropy:
                distillation_loss, distillation_grad = cross_entropy_forward_backward(
                    logits.flatten(0, -2),
                    distillation_target,
                    loss_mask,
                    group=self._tensor_space.distributed.tensor_group if self._parallel_embeddings else None,
                    grad_output=grad_output * self._loss_coefficient * self._distillation_loss_factor,
                    implementation=self._cross_entropy_impl,
                    logits_scale_factor=self._logits_scale_factor,
                    target_format=TargetFormat.logits,
                )
            else:
                raise ValueError(f"Invalid distillation loss implementation: {self._distillation_loss_implementation}")
            distillation_loss = distillation_loss * self._distillation_loss_factor
        else:
            distillation_loss, distillation_grad = None, None

        # TODO: de-allocate earlier.
        del logits

        # TODO: Accumulate grads in-place to reduce memory and compute overhead.
        grad = _add_tensors(dpo_grad, lm_grad, distillation_grad)

        # TODO: Return individual losses?
        loss = _add_tensors(dpo_loss, lm_loss, distillation_loss)
        if self.training and losses is not None:
            if dpo_loss is not None:
                losses[LanguageModelLossNames.dpo_loss].append(dpo_loss.detach())
            if self._config.distillation_model is not None and distillation_loss is not None:
                losses[LanguageModelLossNames.distillation_loss].append(distillation_loss.detach())
            if self._config.distillation_model is not None and lm_loss is not None:
                losses[LanguageModelLossNames.distil_lm_loss].append(lm_loss.detach())

        return loss, output_parallel_linear_backward(grad, context) if self.training else None


def _add_tensors(*tensors: torch.Tensor | None) -> torch.Tensor:
    tensors = [tensor for tensor in tensors if tensor is not None]
    if len(tensors) > 1:
        return sum(tensors)
    elif len(tensors) == 1:
        return tensors[0]
    else:
        raise RuntimeError()
