import collections
import dataclasses
import typing

import pytest
import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.language_model.config import LM_HEAD_LOSS_NAME, LanguageModelKwargs
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.language_model.loss.config import LanguageModelLossKwargs
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.utils import Assert
from tests.layers.test_lm_losses import reference_grpo_loss
from tests.utils.utils import get_base_model, get_stage

NUM_TOKENS = 200
HIDDEN_SIZE = 256
VOCAB_SIZE = 500


@dataclasses.dataclass
class LMHeadTestConfig:
    name: str
    label_loss: bool | float = False
    distillation_loss: bool | float = False
    z_loss: bool | float = False
    grpo_loss: bool | float = False
    logits_scale_factor: float = 1.0
    compute_dtype: DataType = DataType.float32
    full_precision_residual: bool = False
    loss_masking: bool = False
    prediction_heads: int = 1
    tied_embedding_weight: bool = False
    num_splits: int = 1

    @property
    def actual_label_loss(self):
        return (
            True
            if self.label_loss is False
            and self.distillation_loss is False
            and self.z_loss is False
            and self.grpo_loss is False
            else self.label_loss
        )

    def get_config(self) -> GPTModelConfig:
        head_config = {
            "normalization": {"type": "rms_norm"},
            "logits_scale_factor": self.logits_scale_factor,
            "cross_entropy_splits": self.num_splits,
            "prediction_heads": self.prediction_heads,
        }
        losses = {}
        if self.label_loss is not False:
            losses["label"] = {"type": "label"}
            if isinstance(self.label_loss, float):
                losses["label"]["weight"] = self.label_loss
        if self.distillation_loss is not False:
            losses["distillation"] = {"type": "distillation", "reference_model": "distillation"}
            if isinstance(self.distillation_loss, float):
                losses["distillation"]["weight"] = self.distillation_loss
        if self.z_loss is not False:
            losses["z_loss"] = {"type": "z_loss"}
            if isinstance(self.z_loss, float):
                losses["z_loss"]["weight"] = self.z_loss
        if self.grpo_loss is not False:
            losses["grpo_loss"] = {"type": "grpo"}
            if isinstance(self.grpo_loss, float):
                losses["grpo_loss"]["weight"] = self.grpo_loss
        if losses:
            head_config["losses"] = losses

        return GPTModelConfig.from_dict(
            {
                "base_model": {
                    "decoder": {"num_blocks": 0},
                    "embeddings": {"vocab_size": VOCAB_SIZE, "full_precision_residual": self.full_precision_residual},
                    "head": head_config,
                    "hidden_size": HIDDEN_SIZE,
                    "tied_embedding_weight": self.tied_embedding_weight,
                },
                "distributed": {"compute_dtype": self.compute_dtype, "use_cuda": torch.cuda.is_available()},
            },
        )

    def get_inputs(self) -> tuple[torch.Tensor, dict[str, typing.Any]]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ = torch.randn(
            (NUM_TOKENS, HIDDEN_SIZE),
            dtype=(torch.float32 if self.full_precision_residual else self.compute_dtype.torch),
            device=device,
            requires_grad=True,
        )
        kwargs: dict[str, typing.Any] = {
            AttentionKwargs.grad_output: 1.0,
        }
        if self.loss_masking:
            kwargs[LanguageModelKwargs.loss_mask] = [
                torch.randint(0, 2, (NUM_TOKENS,), dtype=torch.bool, device=device)
                for _ in range(self.prediction_heads)
            ]
        if self.actual_label_loss is not False or self.grpo_loss is not False:
            labels = [
                torch.randint(
                    0,
                    VOCAB_SIZE,
                    (NUM_TOKENS,),
                    dtype=torch.int64,
                    device=device,
                )
                for _ in range(self.prediction_heads)
            ]
            if LanguageModelKwargs.loss_mask in kwargs:
                labels = [
                    torch.where(mask, labels_, -100)
                    for labels_, mask in zip(labels, kwargs[LanguageModelKwargs.loss_mask], strict=True)
                ]
            kwargs[LanguageModelKwargs.labels] = labels

        if self.distillation_loss is not False:
            assert self.prediction_heads == 1
            kwargs[f"reference_distillation_hidden_states"] = {
                "head.logits": torch.randn(
                    input_.shape[:-1] + (VOCAB_SIZE,),
                    dtype=input_.dtype,
                    device=device,
                )
            }
        if self.grpo_loss is not False:
            kwargs[LanguageModelLossKwargs.advantages] = [
                torch.randn(input_.shape[:-1], dtype=torch.float32, device=device)
                for _ in range(self.prediction_heads)
            ]
            kwargs[LanguageModelLossKwargs.old_log_probabilities] = [
                torch.randn(input_.shape[:-1], dtype=torch.float32, device=device)
                for _ in range(self.prediction_heads)
            ]
            kwargs[LanguageModelLossKwargs.label_counts] = [
                torch.full(input_.shape[:-1], float((labels_ >= 0).sum()), dtype=torch.float32, device=device)
                for labels_ in kwargs[LanguageModelKwargs.labels]
            ]
        return input_, kwargs

    def get_reference_outputs(
        self,
        head: LanguageModelHead,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        tied_logit_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # Get reference outputs and grads
        logit_weight = (
            (head.output_weights if tied_logit_weight is None else tied_logit_weight).detach().requires_grad_()
        )
        normalization_weight = head.final_norm.weight.detach().requires_grad_()
        input_ = input_.detach().requires_grad_()

        hidden = torch.rms_norm(input_.to(normalization_weight.dtype), input_.shape[-1:], normalization_weight, 1e-5)
        logits = torch.nn.functional.linear(hidden, logit_weight).float()

        if self.logits_scale_factor is not None:
            logits = logits * self.logits_scale_factor

        names_losses_weights = []

        if self.actual_label_loss is not False or self.grpo_loss is not False:
            labels = kwargs[LanguageModelKwargs.labels][head._prediction_distance - 1]

        if self.actual_label_loss is not False:
            label_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none").mean()
            names_losses_weights.append(("label", label_loss, float(self.actual_label_loss)))
            # total_loss = total_loss + float(self.actual_label_loss) * label_loss

        if self.distillation_loss is not False:
            distillation_loss = torch.nn.functional.cross_entropy(
                logits,
                torch.softmax(kwargs[f"reference_distillation_hidden_states"]["head.logits"], -1),
                reduction="none",
            )
            if LanguageModelKwargs.loss_mask in kwargs:
                distillation_loss = (
                    distillation_loss * kwargs[LanguageModelKwargs.loss_mask][head._prediction_distance - 1]
                )
            distillation_loss = distillation_loss.mean()
            names_losses_weights.append(("distillation", distillation_loss, float(self.distillation_loss)))

        if self.z_loss is not False:
            z_loss = torch.logsumexp(logits, dim=-1) ** 2
            if LanguageModelKwargs.loss_mask in kwargs:
                z_loss = z_loss * kwargs[LanguageModelKwargs.loss_mask][head._prediction_distance - 1]
            z_loss = z_loss.mean()
            names_losses_weights.append(("z_loss", z_loss, float(self.z_loss)))

        if self.grpo_loss is not False:
            grpo_loss, new_logprobs = reference_grpo_loss(
                logits,
                labels,
                kwargs[LanguageModelLossKwargs.advantages][head._prediction_distance - 1],
                kwargs[LanguageModelLossKwargs.old_log_probabilities][head._prediction_distance - 1],
            )
            names_losses_weights.append(("grpo_loss", grpo_loss, float(self.grpo_loss)))
            names_losses_weights.append(("grpo_loss_new_logprobs", new_logprobs, 0.0))

        actual_losses = [loss * weight for _, loss, weight in names_losses_weights if weight != 0.0]
        total_loss = sum(actual_losses)
        total_loss.backward()
        losses = {LM_HEAD_LOSS_NAME: total_loss.detach()} | {
            name: loss.detach()
            for name, loss, weight in names_losses_weights
            if weight != 1.0 or len(actual_losses) > 1
        }

        if head._prediction_distance > 1:
            losses = {f"{name}_{head._prediction_distance}": loss for name, loss in losses.items()}

        return total_loss.detach(), input_.grad, logit_weight.grad, normalization_weight.grad, losses


_lm_head_test_configs = []


def _add_configs(base_name: str, **kwargs):
    # Loss masking and splits are important and error-prone, so we test them for all scenarios.
    for loss_masking in (False, True):
        for num_splits in (1, 2):
            _lm_head_test_configs.append(
                LMHeadTestConfig(
                    f"{base_name}{"_masked" if loss_masking else ""}{"" if num_splits == 1 else "_split"}",
                    loss_masking=loss_masking,
                    num_splits=num_splits,
                    **kwargs,
                )
            )


_add_configs("default")
_add_configs("bfloat16", compute_dtype=DataType.bfloat16)
_add_configs("full_precision_residual", full_precision_residual=True)
_add_configs("logit_scaling", logits_scale_factor=5.0)
_add_configs("tied_embedding_weight", tied_embedding_weight=True)
_add_configs("multi_token_prediction", prediction_heads=2)
_add_configs("label_loss", label_loss=True)
_add_configs("distillation_loss", distillation_loss=True)
_add_configs("z_loss", z_loss=True)
_add_configs("grpo_loss", grpo_loss=True)
_add_configs("label_and_distillation_loss", label_loss=True, distillation_loss=True)
_add_configs("label_and_z_loss_weighted", label_loss=True, z_loss=0.5)
_add_configs("label_and_distillation_loss_zero_weight", label_loss=True, distillation_loss=0.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(_lm_head_test_config, id=_lm_head_test_config.name)
        for _lm_head_test_config in _lm_head_test_configs
    ],
)
def test_lm_head(test_config: LMHeadTestConfig):
    model_config = test_config.get_config()
    model, distributed = get_base_model(model_config)
    input_, kwargs = test_config.get_inputs()

    tied_logit_weight = (
        torch.nn.Parameter(
            torch.empty(
                VOCAB_SIZE, HIDDEN_SIZE, dtype=distributed.config.compute_dtype.torch, device=distributed.device
            ).normal_(HIDDEN_SIZE**-0.5)
        )
        if test_config.tied_embedding_weight or test_config.prediction_heads > 1
        else None
    )

    for prediction_distance in range(1, model_config.base_model.head.prediction_heads + 1):
        # Prepare the LM head
        head: LanguageModelHead = (
            model.head if prediction_distance == 1 else model.multi_token_prediction.heads[prediction_distance - 2]
        )
        Assert.custom(isinstance, head, LanguageModelHead)
        Assert.eq(head._prediction_distance, prediction_distance)
        is_duplicate = test_config.tied_embedding_weight or prediction_distance > 1
        stage = get_stage(
            [head],
            distributed,
            tied_parameter_duplicates=[head.output_weights.tensor_name] if is_duplicate else [],
            tied_parameter_duplicate_buffers=(
                {head.output_weights.tensor_name: tied_logit_weight} if is_duplicate else {}
            ),
            # Names must be kept as-is for tied weights.
            set_names=False,
        )

        ref_total_loss, ref_input_grad, ref_logit_weight_grad, ref_normalization_weight_grad, ref_losses = (
            test_config.get_reference_outputs(
                head, input_, kwargs, tied_logit_weight if prediction_distance > 1 else None
            )
        )

        # Prepare LM head inputs
        if head._is_last_head:
            head_input = input_.detach().requires_grad_()
            output_grad = input_.new_full((), float("nan"))
        else:
            shared_hidden = torch.randn_like(input_)
            head_input = torch.stack((shared_hidden, input_.detach())).requires_grad_()
            output_grad = torch.randn_like(shared_hidden)

        if is_duplicate:
            logit_weight = tied_logit_weight
            logit_weight.grad_buffer = torch.full_like(logit_weight, float("nan"))
            logit_weight.param_grad_is_zero = True
        else:
            logit_weight = head.output_weights

        losses = collections.defaultdict(list)
        output, context = stage.forward(head_input, kwargs, losses)
        print(losses)
        stage.backward(output_grad, context)
        threshold = 1e-5 if distributed.config.compute_dtype == DataType.float32 else 5e-3
        min_threshold = (
            1e-5 if distributed.config.compute_dtype == DataType.float32 else 1e-4
        ) * test_config.logits_scale_factor

        loss_definitions_ = head.get_loss_definitions()
        loss_definitions = {definition.name: definition for definition in loss_definitions_}
        Assert.eq(len(loss_definitions), len(loss_definitions_))
        Assert.eq(losses.keys(), ref_losses.keys(), loss_definitions.keys())

        losses = {name: loss[0] if len(loss) == 1 else torch.stack(loss).sum() for name, loss in losses.items()}
        losses = {
            name: loss_definition.reduce(losses[name], distributed)
            for name, loss_definition in loss_definitions.items()
        }

        for name, loss in losses.items():
            Assert.rms_close_relative(loss, ref_losses[name], threshold, min_threshold, msg=name)

        if head._is_last_head:
            # Assert.all_equal(output, losses[lm_head_loss_name][0])
            input_grad = head_input.grad
        else:
            Assert.all_equal(output, shared_hidden)
            shared_hidden_grad, input_grad = head_input.grad.unbind()
            Assert.all_equal(shared_hidden_grad, output_grad)

        Assert.rms_close_relative(input_grad, ref_input_grad, threshold, min_threshold)
        Assert.rms_close_relative(
            head.final_norm.weight.grad_buffer, ref_normalization_weight_grad, threshold, min_threshold
        )
        Assert.rms_close_relative(logit_weight.grad_buffer, ref_logit_weight_grad, threshold, min_threshold)
