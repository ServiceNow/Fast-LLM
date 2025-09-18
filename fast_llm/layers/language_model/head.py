import logging
import typing

import torch
from torch._C._distributed_c10d import ReduceOp  # noqa
from torch.distributed import all_reduce

from fast_llm.core.ops import split_op
from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.autograd import grad_is_context, wrap_forward_backward
from fast_llm.functional.config import CrossEntropyImpl, DistillationLossImpl, TargetFormat, TritonConfig
from fast_llm.functional.cross_entropy import cross_entropy_forward_backward, reverse_kl_forward_backward
from fast_llm.functional.dpo import compute_dpo_loss
from fast_llm.functional.linear import output_parallel_linear_backward, output_parallel_linear_forward
from fast_llm.layers.block.block import BlockLayerBase
from fast_llm.layers.block.config import BlockDimNames
from fast_llm.layers.common.auxiliary_loss import AuxiliaryLoss, z_loss
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import (
    LanguageModelEmbeddingsConfig,
    LanguageModelHeadConfig,
    LanguageModelKwargs,
    LanguageModelLossNames,
)
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert, div, get_unique

logger = logging.getLogger(__name__)

OUTPUT_WEIGHTS = "output_weights"


class LanguageModelHead[ConfigType: LanguageModelHeadConfig](BlockLayerBase[ConfigType], Layer):
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
        prediction_distance: int,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )
        self._vocab_parallel = self._distributed_config.tensor_parallel > 1 and embeddings_config.vocab_parallel
        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        self._sequence_parallel_logits = self._sequence_parallel and not self._vocab_parallel
        if self._config.cross_entropy_splits is not None and self._sequence_parallel:
            assert not self._vocab_parallel

        self._loss_coefficient = (
            self._config.prediction_loss_coefficient[prediction_distance]
            if self._config.prediction_loss_coefficient
            else 1.0
        )
        self._loss_name = LanguageModelLossNames.multi_token_prediction_loss(prediction_distance)

        # Distance of the target token prediction
        # 0: next-token prediction
        # >0: multi-token prediction (MTP)
        Assert.geq(prediction_distance, 0)
        self._prediction_distance = prediction_distance
        self._is_last_head = self._prediction_distance == self._config.prediction_heads - 1

        if not self._config.enable_dpo:
            self._cross_entropy_impl = self._config.cross_entropy_implementation
            if self._cross_entropy_impl == CrossEntropyImpl.auto:
                if self._vocab_parallel:
                    self._cross_entropy_impl = CrossEntropyImpl.fused
                elif TritonConfig.TRITON_ENABLED:
                    self._cross_entropy_impl = CrossEntropyImpl.triton
                else:
                    self._cross_entropy_impl = CrossEntropyImpl.fused

        self._forward = wrap_forward_backward(self._forward_backward, grad_is_context)

        self.final_norm = self._config.normalization.get_layer(
            self._hidden_dim, lr_scale=self._lr_scale, peft=self._peft
        )

        self._vocab_dim = TensorDim(
            "vocab", embeddings_config.vocab_size, self._parallel_dim if self._vocab_parallel else None
        )
        # Only the first head defines the output weights
        if self._prediction_distance == 0 and not self._config.tied_weight:
            # untie embedding weights
            self.output_weights = self._config.output_weight.get_parameter(
                (self._vocab_dim, self._hidden_dim),
                default_initialization=init_normal_(std=self._hidden_size**-0.5),
                lr_scale=self._lr_scale,
                peft=self._peft,
            )

    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            if self._is_last_head:
                return TensorMeta.from_dims(
                    (scalar_dim,),
                    tensor_name="Loss",
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
        input_ = input_.detach().requires_grad_(do_grad := targets is not None and self.training)
        with torch.enable_grad():
            ln_output = self.final_norm(input_)

            if "output_hidden_states" in kwargs and kwargs["output_hidden_states"]:
                # The last hidden layer output is returned normalized in the HF Transformers-style output, at least for LLama style models.
                # So, if needed, we gather the data after normalization and set it as the output of the previous layer.
                dims = list(kwargs[LanguageModelKwargs.hidden_dims])
                sequence_index = 1 - int(kwargs[LanguageModelKwargs.sequence_first])
                dims[sequence_index] = (
                    TensorDim(
                        BlockDimNames.sequence_q_tp,
                        dims[sequence_index].global_size,
                        self._distributed_config.get_distributed_dim(DistributedDimNames.tensor),
                    )
                    if self._sequence_parallel_logits
                    else TensorDim(BlockDimNames.sequence_q, dims[sequence_index].global_size)
                )
                meta = TensorMeta.from_dims(tuple(dims), tensor_name="transformer hidden_state", dtype=ln_output.dtype)
                hidden_state, _ = meta.local_to_global(ln_output.detach())
                kwargs["hidden_states"][len(kwargs["hidden_states"]) - 1]["tensor"] = hidden_state

        grad_output = kwargs[LanguageModelKwargs.grad_output] / (
            self._parallel_dim.size if self._sequence_parallel_logits else 1
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
        if self._config.enable_dpo:
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

            if self._config.distillation_model is None or self._config.language_model_loss_factor > 0.0:
                lm_target = kwargs.get(LanguageModelKwargs.labels)
                if lm_target is not None:
                    # MTP: Shift the labels
                    lm_target_sequence_length = (
                        lm_target.size(1 - kwargs[LanguageModelKwargs.sequence_first])
                        + 1
                        - self._config.prediction_heads
                    )
                    if LanguageModelKwargs.sequence_q_dim in kwargs:
                        Assert.eq(lm_target_sequence_length, kwargs[LanguageModelKwargs.sequence_q_dim].size)
                    lm_target_slice = slice(
                        self._prediction_distance, self._prediction_distance + lm_target_sequence_length
                    )
                    lm_target = (
                        lm_target[lm_target_slice]
                        if kwargs[LanguageModelKwargs.sequence_first]
                        else lm_target[:, lm_target_slice]
                    ).flatten()
            else:
                lm_target = None

        targets = (dpo_target, lm_target, distillation_target, loss_mask)
        if self._sequence_parallel_logits:
            targets = [None if target is None else split_op(target, self._parallel_dim.group, 0) for target in targets]
        if not any(target is not None for target in targets):
            # Simplify so we don't have to check every time.
            targets = None
        return targets

    def _get_output_weights(self, kwargs: dict) -> torch.Tensor:
        if self._config.tied_weight:
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
        if self._config.cross_entropy_splits is None or targets is None:
            loss, logit_input_grad = self._logits_cross_entropy_forward_backward(
                input_, targets, weight, grad_output, kwargs, losses
            )
            if targets is None:
                # TODO: Make a proper way of returning the model output.
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
                get_unique(target.size(0) for target in targets if target is not None),
                self._config.cross_entropy_splits,
            )
            tensors_split = [
                [None] * self._config.cross_entropy_splits if tensor is None else tensor.split(split_size)
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
        loss_count = (self._config.cross_entropy_splits or 1) * (
            self._parallel_dim.size if self._sequence_parallel_logits else 1
        )
        if loss_count != 1:
            loss.div_(loss_count)
        if self._sequence_parallel_logits:
            # TODO: Async
            all_reduce(loss, group=self._parallel_dim.group)
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
        group = self._parallel_dim.group if self._vocab_parallel else None
        logits, context = output_parallel_linear_forward(
            input_=input_,
            weight=weight,
            bias=None,
            group=group,
            sequence_parallel=self._sequence_parallel and self._vocab_parallel,
        )

        if self._config.logit_z_loss > 0.0:
            logits = z_loss(
                logits,
                self._config.logit_z_loss,
                self.training,
                grad_output,
                losses,
                LanguageModelLossNames.z_loss,
                logits_scale_factor=self._config.logits_scale_factor,
            )
        if self._debug.enabled and self._config.cross_entropy_splits is None:
            sequence_dim = BlockDimNames.sequence_q_tp if self._sequence_parallel_logits else BlockDimNames.sequence_q
            batch_dim = kwargs[LanguageModelKwargs.hidden_dims][1 if kwargs[LanguageModelKwargs.sequence_first] else 0]
            dims = (
                (sequence_dim, batch_dim, self._vocab_dim)
                if kwargs[LanguageModelKwargs.sequence_first]
                else (batch_dim, sequence_dim, self._vocab_dim)
            )
            self._debug(logits, "Language model logits", dims, kwargs, scale=self._config.logits_scale_factor)

        if targets is None:
            return logits * self._config.logits_scale_factor, None
        dpo_target, lm_target, distillation_target, loss_mask = targets

        if dpo_target is not None:
            dpo_loss, dpo_grad = compute_dpo_loss(
                logits,
                dpo_target,
                kwargs.get(f"{self._config.dpo_reference_model}_logits"),
                kwargs[LanguageModelKwargs.chosen_spans],
                kwargs[LanguageModelKwargs.rejected_spans],
                self._config.dpo_beta,
                grad_output * self._loss_coefficient,
            )
        else:
            dpo_loss, dpo_grad = None, None

        if lm_target is not None:
            lm_loss, lm_grad = cross_entropy_forward_backward(
                logits.flatten(0, -2),
                lm_target,
                None,
                group=group,
                grad_output=grad_output * self._loss_coefficient * self._config.language_model_loss_factor,
                implementation=self._cross_entropy_impl,
                logits_scale_factor=self._config.logits_scale_factor,
                target_format=TargetFormat.labels,
            )
            lm_loss = lm_loss * self._config.language_model_loss_factor
        else:
            lm_loss, lm_grad = None, None

        if distillation_target is not None and self._config.distillation_loss_factor > 0.0:
            if self._config.distillation_loss_implementation == DistillationLossImpl.reverse_kl:
                distillation_loss, distillation_grad = reverse_kl_forward_backward(
                    logits.flatten(0, -2),
                    distillation_target,
                    loss_mask,
                    grad_output=grad_output * self._loss_coefficient * self._config.distillation_loss_factor,
                    group=group,
                    logits_scale_factor=self._config.logits_scale_factor,
                    teacher_softmax_temperature=self._config.teacher_softmax_temperature,
                    target_format=(
                        TargetFormat.labels if self._config.distillation_model is None else TargetFormat.logits
                    ),
                )
            elif self._config.distillation_loss_implementation == DistillationLossImpl.cross_entropy:
                distillation_loss, distillation_grad = cross_entropy_forward_backward(
                    logits.flatten(0, -2),
                    distillation_target,
                    loss_mask,
                    group=group,
                    grad_output=grad_output * self._loss_coefficient * self._config.distillation_loss_factor,
                    implementation=self._cross_entropy_impl,
                    logits_scale_factor=self._config.logits_scale_factor,
                    target_format=TargetFormat.logits,
                )
            else:
                raise ValueError(
                    f"Invalid distillation loss implementation: {self._config.distillation_loss_implementation}"
                )
            distillation_loss = distillation_loss * self._config.distillation_loss_factor
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
