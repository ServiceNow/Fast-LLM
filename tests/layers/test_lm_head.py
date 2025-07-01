import typing

import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.functional.config import CrossEntropyImpl
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.layers.language_model.head import OUTPUT_WEIGHTS, LanguageModelHead
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.utils import Assert
from tests.utils.utils import get_base_model, get_stage, requires_cuda


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
):
    hidden = torch.rms_norm(
        input_.to(rms_weight.dtype),
        input_.shape[-1:],
        rms_weight,
        1e-5,
    )
    logits = torch.nn.functional.linear(hidden, logit_weight).float()
    if logit_scale_factor != 1.0:
        logits *= logit_scale_factor
    z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2) if logit_z_loss > 0 else None
    if target.ndim == logits.ndim:
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, -2), target.float().softmax(-1).flatten(0, -2), reduction="none"
        )
        if loss_mask is not None:
            loss = loss * loss_mask.flatten()
        loss = loss.mean()
    else:
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, -2), target.flatten())
    loss.backward(torch.full_like(loss, grad_output))
    return loss, z_loss


SEQUENCE_LENGTH = 200
BATCH_SIZE = 4
HIDDEN_SIZE = 256
VOCAB_SIZE = 500


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("cross_entropy_impl", tuple(CrossEntropyImpl))
@pytest.mark.parametrize(
    ("config_dict", "distributed_config_dict", "loss_masking"),
    (
        ({}, {}, False),
        ({}, {"training_dtype": DataType.bfloat16}, False),
        ({"transformer": {"full_precision_residual": True}}, {"training_dtype": DataType.bfloat16}, False),
        ({"sequence_first": True}, {}, False),
        ({"logit_z_loss": 1e-3}, {}, False),
        ({"logits_scale_factor": 5.0}, {}, False),
        ({"tie_word_embeddings": False}, {}, False),
        ({"prediction_heads": 2}, {}, False),
        ({}, {}, True),
        ({"distillation_model": "distillation"}, {}, False),
        ({"distillation_model": "distillation"}, {}, True),
    ),
)
def test_lm_head(
    cross_entropy_impl: CrossEntropyImpl,
    config_dict: dict[str, typing.Any],
    distributed_config_dict: dict[str, typing.Any],
    loss_masking: bool,
):
    config = GPTBaseModelConfig.from_dict(
        {
            "transformer": {
                "normalization": {"type": "rms_norm"},
                "hidden_size": HIDDEN_SIZE,
                "num_layers": 0,
            },
            "vocab_size": VOCAB_SIZE,
            "cross_entropy_impl": cross_entropy_impl,
        },
        config_dict,
        update_type=UpdateType.update,
    )

    model, distributed = get_base_model(
        GPTModelConfig.from_dict(
            {
                "base_model": config,
                "distributed": distributed_config_dict,
            },
        )
    )

    sequence_first = config.sequence_first or (
        config.cross_entropy_splits is not None and config.cross_entropy_splits > 1
    )
    input_ = torch.randn(
        (SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE) if sequence_first else (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE),
        dtype=(
            distributed.config.optimization_dtype.torch
            if config.transformer.full_precision_residual
            else distributed.config.training_dtype.torch
        ),
        device=distributed.device,
        requires_grad=True,
    )
    label_shape = (
        (SEQUENCE_LENGTH + config.prediction_heads - 1, BATCH_SIZE)
        if sequence_first
        else (BATCH_SIZE, SEQUENCE_LENGTH + config.prediction_heads - 1)
    )
    if loss_masking:
        loss_mask = torch.randint(0, 2, label_shape, dtype=torch.bool, device=distributed.device)
    else:
        loss_mask = None
    kwargs = {
        TransformerKwargs.sequence_first: sequence_first,
        TransformerKwargs.grad_output: 1.0,
    }
    if config.distillation_model is None:
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
    else:
        assert config.prediction_heads == 1
        target = torch.randn(
            input_.shape[:-1] + (VOCAB_SIZE,),
            dtype=input_.dtype,
            device=distributed.device,
        )
        kwargs[f"{config.distillation_model}_logits"] = target
        if loss_mask is not None:
            kwargs[LanguageModelKwargs.loss_mask] = loss_mask

    if config.tie_word_embeddings or config.prediction_heads > 1:
        logit_weight = (
            torch.empty(
                VOCAB_SIZE, HIDDEN_SIZE, dtype=distributed.config.training_dtype.torch, device=distributed.device
            )
            .normal_(config.transformer.init_method_std)
            .requires_grad_(True)
        )
        kwargs[WORD_EMBEDDINGS_WEIGHT if config.tie_word_embeddings else OUTPUT_WEIGHTS] = logit_weight
    else:
        logit_weight = None

    for prediction_distance, layer_index in enumerate(model.model_head_indices):
        # Prepare the LM head
        head: LanguageModelHead = model[layer_index]
        Assert.custom(isinstance, head, LanguageModelHead)
        Assert.eq(head._prediction_distance, prediction_distance)
        stage = get_stage([head], distributed)

        # Get reference outputs and grads
        if logit_weight is None:
            logit_weight = head.output_weights
        else:
            logit_weight.grad_buffer = torch.full_like(logit_weight, float("nan"))
            logit_weight.param_grad_is_zero = True

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
            logit_scale_factor=config.logits_scale_factor,
            logit_z_loss=config.logit_z_loss,
        )

        # Prepare LM head inputs
        if head._is_last_head:
            head_input = input_
            output_grad = ref_input.new_full((), float("nan"))
        else:
            shared_hidden = torch.randn_like(input_)
            head_input = torch.stack((shared_hidden, input_.detach())).requires_grad_()
            output_grad = torch.randn_like(shared_hidden)

        loss_name = f"language_model_loss_{prediction_distance}" if prediction_distance > 0 else "language_model_loss"
        Assert.eq(head._loss_name, loss_name)
        loss_keys = {loss_name}
        if ref_z_loss is not None:
            loss_keys.add("z_loss")
        losses = {key: [] for key in loss_keys}
        output, context = stage.forward(head_input, kwargs, losses)
        stage.backward(output_grad, context)

        threshold = 1e-5 if distributed.config.training_dtype == DataType.float32 else 5e-3
        min_threshold = (
            1e-5 if distributed.config.training_dtype == DataType.float32 else 1e-4
        ) * config.logits_scale_factor

        Assert.eq(losses.keys(), loss_keys)
        Assert.eq(len(losses[loss_name]), 1)
        if ref_z_loss is not None:
            Assert.eq(len(losses["z_loss"]), 1)
            Assert.rms_close_relative(losses["z_loss"][0], ref_z_loss, threshold, min_threshold)

        Assert.rms_close_relative(losses[loss_name][0], ref_loss, threshold, min_threshold)

        if head._is_last_head:
            Assert.all_equal(output, losses[loss_name][0])
            input_grad = head_input.grad
        else:
            Assert.all_equal(output, shared_hidden)
            shared_hidden_grad, input_grad = head_input.grad.unbind()
            Assert.all_equal(shared_hidden_grad, output_grad)

        Assert.rms_close_relative(input_grad, ref_input.grad, threshold, min_threshold)
        Assert.rms_close_relative(head.final_norm.weight.grad_buffer, ref_rms_weight.grad, threshold, min_threshold)
        Assert.rms_close_relative(logit_weight.grad_buffer, ref_logit_weight.grad, threshold, min_threshold)
