import typing

import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.config import CrossEntropyImpl
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.language_model.config import LanguageModelHeadConfig
from fast_llm.layers.language_model.head import LanguageModelHead
from fast_llm.layers.language_model.kwargs import LanguageModelKwargs
from fast_llm.layers.language_model.lm_head_losses import LanguageModelLossConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_base_model, get_stage, requires_cuda


def _reverse_kl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    teacher_softmax_temperature: float = 1.0,
):
    scaled_target = torch.clamp(target / teacher_softmax_temperature, min=-50, max=50)
    teacher_log_probs = torch.log_softmax(scaled_target, dim=-1)

    with torch.enable_grad():
        # Use log_softmax for consistency instead of _fused_softmax
        logits = torch.clamp(logits, min=-50, max=50)
        student_log_probs = torch.log_softmax(logits, dim=-1)
        if loss_mask is None:
            loss = torch.nn.functional.kl_div(
                teacher_log_probs,  # input = log(p)
                student_log_probs,  # target = log(q)
                reduction="batchmean",
                log_target=True,
            )
        else:
            # Apply loss mask - this requires some reshaping
            loss_per_sample = torch.nn.functional.kl_div(
                teacher_log_probs, student_log_probs, reduction="none", log_target=True
            ).sum(dim=-1)
            loss = (loss_per_sample * loss_mask.flatten()).sum() / loss_mask.sum()
    return loss


def _kl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    teacher_softmax_temperature: float = 1.0,
):
    return _reverse_kl_loss(
        target,
        logits,
        loss_mask,
        teacher_softmax_temperature,
    )


def _lm_head(
    input_: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
    *,
    # config:LanguageModelBaseConfig,
    rms_weight: torch.Tensor,
    logit_weight: torch.Tensor,
    grad_output: float = 1.0,
    logit_scale_factor: float = 1.0,
    logit_z_loss=0.0,
    losses: dict[str, LanguageModelLossConfig],
):
    hidden = torch.rms_norm(
        input_.to(rms_weight.dtype),
        input_.shape[-1:],
        rms_weight,
        1e-5,
    )
    logits = torch.nn.functional.linear(hidden, logit_weight).float()

    if "dist_loss" in losses:
        if losses["dist_loss"].type == "reverse_kl_distillation":
            Assert.eq(logits.shape, target.shape)
            loss = _reverse_kl_loss(
                (logits * logit_scale_factor).flatten(0, -2), (target * logit_scale_factor).flatten(0, -2), loss_mask
            )
            # Apply distillation_loss_factor to grad_output for backward
            loss.backward(torch.full_like(loss, grad_output * losses["dist_loss"].weight))
            # Return scaled loss
            return loss * losses["dist_loss"].weight, None
        elif losses["dist_loss"].type == "forward_kl_distillation":
            Assert.eq(logits.shape, target.shape)
            loss = _kl_loss(
                (logits * logit_scale_factor).flatten(0, -2), (target * logit_scale_factor).flatten(0, -2), loss_mask
            )
            # Apply distillation_loss_factor to grad_output for backward
            loss.backward(torch.full_like(loss, grad_output * losses["dist_loss"].weight))
            # Return scaled loss
            return loss * losses["dist_loss"].weight, None

    if logit_scale_factor != 1.0:
        logits *= logit_scale_factor
    z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2) if logit_z_loss > 0 else None
    # Language model loss (cross-entropy with hard labels)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, -2), target.flatten())
    # Apply language_model_loss_factor
    loss.backward(torch.full_like(loss, grad_output * losses["lm_loss"].weight))
    return loss * losses["lm_loss"].weight, z_loss


SEQUENCE_LENGTH = 200
BATCH_SIZE = 4
HIDDEN_SIZE = 256
VOCAB_SIZE = 500


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("cross_entropy_impl", tuple(CrossEntropyImpl))
@pytest.mark.parametrize(
    ("config_dict", "distributed_config_dict", "loss_masking", "prediction_heads"),
    (
        ({}, {}, False, 1),
        ({}, {"compute_dtype": DataType.bfloat16}, False, 1),
        ({"embeddings": {"full_precision_residual": True}}, {"compute_dtype": DataType.bfloat16}, False, 1),
        ({"sequence_first": True}, {}, False, 1),
        ({"head": {"logit_z_loss": 1e-3}}, {}, False, 1),
        ({"head": {"logits_scale_factor": 5.0}}, {}, False, 1),
        ({"tied_embedding_weight": True}, {}, False, 1),
        ({}, {}, False, 2),
        ({}, {}, True, 1),
        # Skip CE distillation for now - not yet implemented in new losses system
        # (
        #     {
        #         "head": {
        #             "distillation_model": "distillation",
        #             "losses": {
        #                 "lm_loss": {
        #                     "type": "cross_entropy",
        #                     "weight_scalor": 0.0,
        #                 },
        #                 "dist_loss": {
        #                     "type": "cross_entropy_dist",  # TODO: Not implemented yet
        #                     "weight_scalor": 1.0,
        #                 }
        #             }
        #         }
        #     },
        #     {},
        #     False,
        #     1,
        # ),
        pytest.param(
            {
                "head": {
                    "losses": {
                        "lm_loss": {
                            "type": "cross_entropy",
                            "weight": 0.0,
                        },
                        "dist_loss": {
                            "type": "reverse_kl_distillation",
                            "weight": 1.0,
                            "distillation_model": "distillation",
                        },
                    },
                }
            },
            {},
            False,
            1,
            id="track_lm_zero_factor",
        ),
        pytest.param(
            {
                "head": {
                    "losses": {
                        "lm_loss": {
                            "type": "cross_entropy",
                            "weight": 0.0,
                        },
                        "dist_loss": {
                            "type": "reverse_kl_distillation",
                            "weight": 0.0,
                            "distillation_model": "distillation",
                        },
                    },
                }
            },
            {},
            False,
            1,
            marks=pytest.mark.xfail(
                reason="Cannot track both losses with zero factor",
                strict=True,
            ),
            id="track_both_zero_factors",
        ),
        pytest.param(
            {
                "head": {
                    "losses": {
                        "lm_loss": {
                            "type": "cross_entropy",
                            "weight": 1.0,
                        },
                        "dist_loss": {
                            "type": "reverse_kl_distillation",
                            "weight": 1.0,
                            "distillation_model": "distillation",
                        },
                    },
                }
            },
            {},
            False,
            1,
            marks=pytest.mark.xfail(
                reason="Cannot track distillation loss without distillation model being set",
                strict=True,
            ),
            id="track_distillation_without_model",
        ),
    ),
)
def test_lm_head(
    cross_entropy_impl: CrossEntropyImpl,
    config_dict: dict[str, typing.Any],
    distributed_config_dict: dict[str, typing.Any],
    loss_masking: bool,
    prediction_heads: int,
):
    head_config = {
        "normalization": {"type": "rms_norm"},
        "losses": {
            "lm_loss": {
                "type": "cross_entropy",
                "implementation": cross_entropy_impl,
                "weight": 1.0,
            }
        },
    }
    config = GPTBaseModelConfig.from_dict(
        {
            "decoder": {"num_blocks": 0},
            "embeddings": {"vocab_size": VOCAB_SIZE},
            "head": (
                head_config
                if prediction_heads == 1
                else {
                    "type": "multi_token_prediction",
                    "head": head_config,
                    "prediction_heads": prediction_heads,
                }
            ),
            "hidden_size": HIDDEN_SIZE,
        },
        config_dict,
        update_type=UpdateType.update,
    )
    head_config: LanguageModelHeadConfig = config.head if prediction_heads == 1 else config.head.head

    model, distributed = get_base_model(
        GPTModelConfig.from_dict(
            {
                "base_model": config,
                "distributed": distributed_config_dict,
            },
        )
    )

    sequence_first = config.sequence_first or (
        head_config.cross_entropy_splits is not None and head_config.cross_entropy_splits > 1
    )
    input_ = torch.randn(
        (SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE) if sequence_first else (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE),
        dtype=(
            distributed.config.optimization_dtype.torch
            if config.embeddings.full_precision_residual
            else distributed.config.compute_dtype.torch
        ),
        device=distributed.device,
        requires_grad=True,
    )
    label_shape = (
        (SEQUENCE_LENGTH + config.head.max_prediction_distance - 1, BATCH_SIZE)
        if sequence_first
        else (BATCH_SIZE, SEQUENCE_LENGTH + config.head.max_prediction_distance - 1)
    )
    if loss_masking:
        loss_mask = torch.randint(0, 2, label_shape, dtype=torch.bool, device=distributed.device)
    else:
        loss_mask = None
    kwargs = {
        AttentionKwargs.sequence_first: sequence_first,
        AttentionKwargs.grad_output: 1.0,
    }
    # always set lm targets
    target = torch.randint(
        0,
        VOCAB_SIZE,
        label_shape,
        dtype=torch.int64,
        device=distributed.device,
    )
    if loss_mask is not None:
        target *= loss_mask

    kwargs[LanguageModelKwargs.labels] = target
    if head_config.distillation_model is not None:
        assert config.head.max_prediction_distance == 1
        target = torch.randn(
            input_.shape[:-1] + (VOCAB_SIZE,),
            dtype=input_.dtype,
            device=distributed.device,
        )
        kwargs[f"{head_config.distillation_model}_logits"] = target
        if loss_mask is not None:
            kwargs[LanguageModelKwargs.loss_mask] = loss_mask

    if config.tied_embedding_weight or config.head.max_prediction_distance > 1:
        logit_weight = torch.nn.Parameter(
            torch.empty(
                VOCAB_SIZE, HIDDEN_SIZE, dtype=distributed.config.compute_dtype.torch, device=distributed.device
            ).normal_(config.hidden_size**-0.5)
        )
    else:
        logit_weight = None

    for prediction_distance, head in enumerate(model.head.heads):
        # Prepare the LM head
        Assert.custom(isinstance, head, LanguageModelHead)
        Assert.eq(head._prediction_distance, prediction_distance)
        is_duplicate = config.tied_embedding_weight or prediction_distance > 0
        stage = get_stage(
            [head],
            distributed,
            tied_parameter_duplicates=[head.output_weights.tensor_name] if is_duplicate else [],
            tied_parameter_duplicate_buffers={head.output_weights.tensor_name: logit_weight} if is_duplicate else {},
            # Names must be kept as-is for tied weights.
            set_names=False,
        )

        # Get reference outputs and grads
        if is_duplicate:
            logit_weight.grad_buffer = torch.full_like(logit_weight, float("nan"))
            logit_weight.param_grad_is_zero = True
        else:
            logit_weight = head.output_weights

        ref_input = input_.detach().requires_grad_()
        ref_rms_weight = head.final_norm.weight.detach().requires_grad_()
        ref_logit_weight = logit_weight.detach().requires_grad_()

        ref_loss, ref_z_loss = _lm_head(
            ref_input,
            (
                target[prediction_distance : prediction_distance + SEQUENCE_LENGTH]
                if sequence_first
                else target[:, prediction_distance : prediction_distance + SEQUENCE_LENGTH]
            ),
            loss_mask,
            rms_weight=ref_rms_weight,
            logit_weight=ref_logit_weight,
            logit_scale_factor=head_config.logits_scale_factor,
            logit_z_loss=head_config.logit_z_loss,
            losses=head_config.losses,
        )

        # Prepare LM head inputs
        if head._is_last_head:
            head_input = input_
            output_grad = ref_input.new_full((), float("nan"))
        else:
            shared_hidden = torch.randn_like(input_)
            head_input = torch.stack((shared_hidden, input_.detach())).requires_grad_()
            output_grad = torch.randn_like(shared_hidden)

        lm_head_loss_name = f"lm_head_loss_{prediction_distance}" if prediction_distance > 0 else "lm_head_loss"
        expected_loss_keys = {lm_head_loss_name}

        # Get expected loss names from the loss configs
        for loss_name, loss_config in head._config.losses.items():
            formatted_name = loss_config.get_formatted_name(loss_name, prediction_distance)
            expected_loss_keys.add(formatted_name)

        if ref_z_loss is not None:
            expected_loss_keys.add(f"z_loss_{prediction_distance}" if prediction_distance > 0 else "z_loss")

        Assert.eq(
            {loss_definition.name: loss_definition.count for loss_definition in head.get_loss_definitions()},
            {loss_key: 1 for loss_key in expected_loss_keys},
        )
        losses = {key: [] for key in expected_loss_keys}
        output, context = stage.forward(head_input, kwargs, losses)
        stage.backward(output_grad, context)

        threshold = 1e-5 if distributed.config.compute_dtype == DataType.float32 else 5e-3
        min_threshold = (
            1e-5 if distributed.config.compute_dtype == DataType.float32 else 1e-4
        ) * head_config.logits_scale_factor

        Assert.eq(losses.keys(), expected_loss_keys)
        Assert.eq(len(losses[lm_head_loss_name]), 1)
        if ref_z_loss is not None:
            Assert.eq(len(losses["z_loss"]), 1)
            Assert.rms_close_relative(losses["z_loss"][0], ref_z_loss, threshold, min_threshold)

        Assert.rms_close_relative(losses[lm_head_loss_name][0], ref_loss, threshold, min_threshold)

        if head._is_last_head:
            Assert.all_equal(output, losses[lm_head_loss_name][0])
            input_grad = head_input.grad
        else:
            Assert.all_equal(output, shared_hidden)
            shared_hidden_grad, input_grad = head_input.grad.unbind()
            Assert.all_equal(shared_hidden_grad, output_grad)

        Assert.rms_close_relative(input_grad, ref_input.grad, threshold, min_threshold)
        Assert.rms_close_relative(head.final_norm.weight.grad_buffer, ref_rms_weight.grad, threshold, min_threshold)
        Assert.rms_close_relative(logit_weight.grad_buffer, ref_logit_weight.grad, threshold, min_threshold)
