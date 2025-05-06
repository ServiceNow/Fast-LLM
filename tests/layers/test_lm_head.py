import typing

import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import StageConfig
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.functional.config import CrossEntropyImpl
from fast_llm.layers.common.config import NormalizationType
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.layers.language_model.head import OUTPUT_WEIGHTS, LanguageModelHead
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig
from fast_llm.models.gpt.model import GPTBaseModel
from fast_llm.utils import Assert
from tests.common import requires_cuda


def _lm_head(
    input_: torch.Tensor,
    target: torch.Tensor,
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
    logits = torch.nn.functional.linear(hidden, logit_weight)
    if logit_scale_factor != 1.0:
        logits *= logit_scale_factor
    z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2) if logit_z_loss > 0 else None
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
    ("config_dict", "distributed_config_dict"),
    (
        ({}, {}),
        ({}, {"training_dtype": DataType.bfloat16}),
        ({"transformer": {"full_precision_residual": True}}, {"training_dtype": DataType.bfloat16}),
        ({"sequence_first": True}, {}),
        ({"logit_z_loss": 1e-3}, {}),
        ({"logits_scale_factor": 5.0}, {}),
        ({"tie_word_embeddings": False}, {}),
        ({"prediction_heads": 2}, {}),
    ),
)
def test_lm_head(
    cross_entropy_impl: CrossEntropyImpl,
    config_dict: dict[str, typing.Any],
    distributed_config_dict: dict[str, typing.Any],
):
    config = GPTBaseModelConfig.from_dict(
        {
            "transformer": {
                "normalization": {"type": NormalizationType.rms_norm},
                "hidden_size": HIDDEN_SIZE,
                "num_layers": 0,
            },
            "vocab_size": VOCAB_SIZE,
            "cross_entropy_impl": cross_entropy_impl,
        },
        config_dict,
        update_type=UpdateType.update,
    )
    distributed_config = DistributedConfig.from_dict(distributed_config_dict)
    distributed = Distributed(distributed_config)
    tensor_space = TensorSpace(distributed_config)
    config.setup_tensor_space(tensor_space)
    tensor_space.setup(distributed)
    model = GPTBaseModel(config, distributed_config)
    model.setup(distributed)

    sequence_first = config.sequence_first or (
        config.cross_entropy_splits is not None and config.cross_entropy_splits > 1
    )
    target = torch.randint(
        0,
        VOCAB_SIZE,
        (
            (SEQUENCE_LENGTH + config.prediction_heads - 1, BATCH_SIZE)
            if sequence_first
            else (BATCH_SIZE, SEQUENCE_LENGTH + config.prediction_heads - 1)
        ),
        dtype=torch.int64,
        device=distributed.device,
    )
    input_ = torch.randn(
        (SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE) if sequence_first else (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE),
        dtype=(
            distributed_config.optimization_dtype.torch
            if config.transformer.full_precision_residual
            else distributed_config.training_dtype.torch
        ),
        device=distributed.device,
        requires_grad=True,
    )
    kwargs = {
        TransformerKwargs.sequence_first: sequence_first,
        LanguageModelKwargs.labels: target,
        TransformerKwargs.grad_output: 1.0,
    }
    if config.tie_word_embeddings or config.prediction_heads > 1:
        logit_weight = (
            torch.empty(
                VOCAB_SIZE, HIDDEN_SIZE, dtype=distributed_config.training_dtype.torch, device=distributed.device
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
        stage = Stage(
            config=StageConfig(),
            base_model=[head],
            distributed_config=distributed_config,
            begin=0,
            end=1,
            index=0,
        )
        stage.setup(distributed=distributed)
        stage.initialize_weights()
        stage.restore_parameters()
        stage.reset_gradients()

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

        threshold = 1e-5 if distributed_config.training_dtype == DataType.float32 else 5e-3
        min_threshold = (
            1e-5 if distributed_config.training_dtype == DataType.float32 else 1e-4
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
