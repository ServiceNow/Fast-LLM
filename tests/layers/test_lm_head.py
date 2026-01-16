import collections
import dataclasses
import typing

import pytest
import torch

from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_base_model, get_stage

SEQUENCE_LENGTH = 200
BATCH_SIZE = 4
HIDDEN_SIZE = 256
VOCAB_SIZE = 500


@dataclasses.dataclass
class LMHeadTestConfig:
    name: str
    label_loss: bool | float = False
    distillation_loss: bool | float = False
    z_loss: bool | float = False
    logits_scale_factor: float = 1.0
    compute_dtype: DataType = DataType.float32
    full_precision_residual: bool = False
    sequence_first: bool = False
    loss_masking: bool = False
    prediction_heads: int = 1
    tied_embedding_weight: bool = False
    cross_entropy_splits: int = 1

    @property
    def actual_label_loss(self):
        return (
            True
            if self.label_loss is False and self.distillation_loss is False and self.z_loss is False
            else self.label_loss
        )

    def get_config(self) -> GPTModelConfig:
        head_config = {
            "normalization": {"type": "rms_norm"},
            "logits_scale_factor": self.logits_scale_factor,
            "cross_entropy_splits": self.cross_entropy_splits,
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
        if losses:
            head_config["losses"] = losses

        return GPTModelConfig.from_dict(
            {
                "base_model": {
                    "decoder": {"num_blocks": 0},
                    "embeddings": {"vocab_size": VOCAB_SIZE, "full_precision_residual": self.full_precision_residual},
                    "head": (
                        head_config
                        if self.prediction_heads == 1
                        else {
                            "type": "multi_token_prediction",
                            "head": head_config,
                            "prediction_heads": self.prediction_heads,
                        }
                    ),
                    "hidden_size": HIDDEN_SIZE,
                    "tied_embedding_weight": self.tied_embedding_weight,
                },
                "distributed": {"compute_dtype": self.compute_dtype, "use_cuda": torch.cuda.is_available()},
            },
        )

    def get_inputs(self) -> tuple[torch.Tensor, dict[str, typing.Any]]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ = torch.randn(
            (
                (SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE)
                if self.sequence_first
                else (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE)
            ),
            dtype=(torch.float32 if self.full_precision_residual else self.compute_dtype.torch),
            device=device,
            requires_grad=True,
        )
        label_shape = (
            (SEQUENCE_LENGTH + self.prediction_heads - 1, BATCH_SIZE)
            if self.sequence_first
            else (BATCH_SIZE, SEQUENCE_LENGTH + self.prediction_heads - 1)
        )
        kwargs: dict[str, typing.Any] = {
            AttentionKwargs.sequence_first: self.sequence_first,
            AttentionKwargs.grad_output: 1.0,
        }
        if self.loss_masking:
            kwargs[LanguageModelKwargs.loss_mask] = torch.randint(0, 2, label_shape, dtype=torch.bool, device=device)
        if self.actual_label_loss is not False:
            labels = torch.randint(
                0,
                VOCAB_SIZE,
                label_shape,
                dtype=torch.int64,
                device=device,
            )
            if LanguageModelKwargs.loss_mask in kwargs:
                labels = torch.where(kwargs[LanguageModelKwargs.loss_mask], -100, labels)
            kwargs[LanguageModelKwargs.labels] = labels

        if self.distillation_loss is not False:
            assert self.prediction_heads == 1
            kwargs[f"distillation_logits"] = torch.randn(
                input_.shape[:-1] + (VOCAB_SIZE,),
                dtype=input_.dtype,
                device=device,
            )
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

        total_loss = 0
        losses = {}

        if self.actual_label_loss is not False:
            if self.sequence_first:
                labels = kwargs[LanguageModelKwargs.labels][
                    head._prediction_distance : head._prediction_distance + logits.size(0)
                ]
            else:
                labels = kwargs[LanguageModelKwargs.labels][
                    :, head._prediction_distance : head._prediction_distance + logits.size(1)
                ]
            label_loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, -2), labels.flatten(), reduction="none"
            ).mean()
            losses["label_loss"] = label_loss.detach()
            total_loss = total_loss + float(self.actual_label_loss) * label_loss

        if self.distillation_loss is not False:
            distillation_loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, -2),
                torch.softmax(kwargs[f"distillation_logits"].flatten(0, -2), -1),
                reduction="none",
            )
            if LanguageModelKwargs.loss_mask in kwargs:
                distillation_loss = distillation_loss * kwargs[LanguageModelKwargs.loss_mask].flatten()
            distillation_loss = distillation_loss.mean()
            losses["distillation_loss"] = distillation_loss.detach()
            total_loss = total_loss + float(self.distillation_loss) * distillation_loss

        if self.z_loss is not False:
            z_loss = torch.logsumexp(logits, dim=-1) ** 2
            if LanguageModelKwargs.loss_mask in kwargs:
                z_loss = z_loss * kwargs[LanguageModelKwargs.loss_mask]
            z_loss = z_loss.mean()
            losses["z_loss"] = z_loss.detach()
            total_loss = total_loss + float(self.z_loss) * z_loss

        total_loss.backward()
        return total_loss.detach(), input_.grad, logit_weight.grad, normalization_weight.grad, losses


_lm_head_test_configs = (
    # TODO: Test DPO loss.
    LMHeadTestConfig("default"),
    LMHeadTestConfig("bfloat16", compute_dtype=DataType.bfloat16),
    LMHeadTestConfig("full_precision_residual", full_precision_residual=True),
    LMHeadTestConfig("sequence_first", sequence_first=True),
    LMHeadTestConfig("logit_scaling", logits_scale_factor=5.0),
    LMHeadTestConfig("tied_embedding_weight", tied_embedding_weight=True),
    LMHeadTestConfig("multi_token_prediction", prediction_heads=2),
    LMHeadTestConfig("cross_entropy_splits", cross_entropy_splits=2, sequence_first=True),
    LMHeadTestConfig("loss_masking", loss_masking=True),
    LMHeadTestConfig("label_loss", label_loss=True),
    LMHeadTestConfig("distillation_loss", distillation_loss=True),
    LMHeadTestConfig("distillation_loss_masked", distillation_loss=True, loss_masking=True),
    LMHeadTestConfig("z_loss", z_loss=True),
    LMHeadTestConfig("z_loss_masked", z_loss=True, loss_masking=True),
    LMHeadTestConfig("label_and_distillation_loss", label_loss=True, distillation_loss=True),
    LMHeadTestConfig("label_and_z_loss_weighted", label_loss=True, z_loss=0.5),
    LMHeadTestConfig("label_and_distillation_loss_zero_weight", label_loss=True, distillation_loss=0.0),
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(_lm_head_test_config, id=_lm_head_test_config.name)
        for _lm_head_test_config in _lm_head_test_configs
    ],
)
def test_lm_head(test_config):
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

    for prediction_distance, head in enumerate(model.head.heads):
        # Prepare the LM head
        Assert.custom(isinstance, head, LanguageModelHead)
        Assert.eq(head._prediction_distance, prediction_distance)
        is_duplicate = test_config.tied_embedding_weight or prediction_distance > 0
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
                head, input_, kwargs, tied_logit_weight if prediction_distance > 0 else None
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

        # lm_head_loss_name = f"lm_head_loss_{prediction_distance}" if prediction_distance > 0 else "lm_head_loss"
        # expected_loss_keys = {lm_head_loss_name}

        ## Get expected loss names from the loss configs
        # for loss_name, loss_config in head._config.losses.items():
        #    formatted_name = loss_config.get_formatted_name(loss_name, prediction_distance)
        #    expected_loss_keys.add(formatted_name)

        # if ref_z_loss is not None:
        #     expected_loss_keys.add(f"z_loss_{prediction_distance}" if prediction_distance > 0 else "z_loss")

        # Assert.eq(
        #    {loss_definition.name: loss_definition.count for loss_definition in head.get_loss_definitions()},
        #    {loss_key: 1 for loss_key in expected_loss_keys},
        # )
        # losses = {key: [] for key in expected_loss_keys}
        losses = collections.defaultdict(list)
        output, context = stage.forward(head_input, kwargs, losses)
        print(losses)
        stage.backward(output_grad, context)
        threshold = 1e-5 if distributed.config.compute_dtype == DataType.float32 else 5e-3
        min_threshold = (
            1e-5 if distributed.config.compute_dtype == DataType.float32 else 1e-4
        ) * test_config.logits_scale_factor

        # Assert.eq(losses.keys(), expected_loss_keys)
        # Assert.eq(len(losses[lm_head_loss_name]), 1)
        # if ref_z_loss is not None:
        #     Assert.eq(len(losses["z_loss"]), 1)
        #     Assert.rms_close_relative(losses["z_loss"][0], ref_z_loss, threshold, min_threshold)

        Assert.rms_close_relative(losses[head._total_head_loss_name][0], ref_total_loss, threshold, min_threshold)

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
