import typing

import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.layers.language_model.head import OUTPUT_WEIGHTS
from fast_llm.layers.transformer.config import TransformerConfig, TransformerKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig
from fast_llm.models.gpt.model import GPTBaseModel
from fast_llm.utils import Assert
from tests.common import requires_cuda

try:
    from fast_llm.layers.ssm.config import SSMConfig
    from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2
    from fast_llm.layers.ssm.mamba_layer import MambaLayer
    from fast_llm.models.ssm.model import HybridSSMBaseModel, HybridSSMBaseModelConfig
except ImportError:
    MambaLayer, HybridSSMBaseModel, HybridSSMBaseModelConfig, DiscreteMamba2 = (
        None,
        None,
        None,
        None,
    )
    # Mamba not installed, skipping tests


run_hybrid_test = MambaLayer is not None and DiscreteMamba2 is not None and torch.cuda.is_available()


SEQUENCE_LENGTH = 200
BATCH_SIZE = 4
HIDDEN_SIZE = 256
VOCAB_SIZE = 500


def materialize_meta_tensors(model, tensor_space):
    # Materialize parameters that are on meta device
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            # Check if the parameter is a custom tensor type
            if hasattr(param, "tensor_name") and hasattr(param, "init_parameter"):
                param_data = param.new_empty(param.shape, device="cuda")
                # Initialize param_data
                param.init_parameter(param_data, tensor_space.distributed)
                # Replace the parameter in the module
                module_path, param_name = name.rsplit(".", 1) if "." in name else (None, name)
                module = model
                if module_path is not None:
                    for part in module_path.split("."):
                        module = getattr(module, part)
                param = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
                # TODO: add param_grad_is_zero etc., grad_buffer, etc., see test_mlp_recomputation
                param.grad = None
                param.grad_buffer = torch.empty_like(param)
                param.param_grad_is_zero = True
                module._parameters[param_name] = param
    return model


@pytest.fixture
def distributed_config():
    return DistributedConfig(
        tensor_parallel=1,
        pipeline_parallel=1,
        sequence_data_parallel=1,
        local_world_size=1,
        world_size=1,
    )


@pytest.fixture
def distributed(distributed_config):
    return Distributed(config=distributed_config)


def get_hybrid_config(hybrid_block_layout=["t", "m"], mtp_heads=["t"]):
    config = HybridSSMBaseModelConfig(
        transformer=TransformerConfig(num_layers=len(hybrid_block_layout)),
        ssm=SSMConfig(),
        hybrid_block_layout=hybrid_block_layout,
        mtp_heads=mtp_heads,
        prediction_heads=len(mtp_heads) + 1,
        init_method_std_embed=0.02,
        init_method_min_embed=-0.02,
        init_method_max_embed=0.02,
        use_position_embeddings=True,
        tie_word_embeddings=False,
    )
    return config


@requires_cuda
@pytest.mark.parametrize(
    "config_dict",
    (
        {"prediction_heads": 1},
        {"prediction_heads": 2, "tie_word_embeddings": False},
        {"prediction_heads": 5, "tie_word_embeddings": False},
    ),
)
def test_transformer_mtp(config_dict: dict[str, typing.Any]):
    config = GPTBaseModelConfig.from_dict(
        {
            "transformer": {
                "hidden_size": HIDDEN_SIZE,
                "num_layers": 2,
            },
            "vocab_size": VOCAB_SIZE,
        },
        config_dict,
        update_type=UpdateType.update,
    )
    distributed_config = DistributedConfig.from_dict({})
    distributed = Distributed(distributed_config)
    model = GPTBaseModel(config, distributed_config)
    model.setup(distributed)
    materialize_meta_tensors(model, model._tensor_space)
    model.to("cuda")

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
    input_ = torch.randint(
        0,
        VOCAB_SIZE,
        (SEQUENCE_LENGTH, BATCH_SIZE) if sequence_first else (BATCH_SIZE, SEQUENCE_LENGTH),
        device=distributed.device,
    )
    attention_mask = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.bool)
    position_ids = torch.arange(SEQUENCE_LENGTH, device="cuda", dtype=torch.int64)
    kwargs = {
        "position_ids": position_ids,
        TransformerKwargs.sequence_first: sequence_first,
        TransformerKwargs.attention_mask: attention_mask,
        TransformerKwargs.attention_mask_value: -100,
        TransformerKwargs.grad_output: 1.0,
        LanguageModelKwargs.labels: target,
    }
    if config.tie_word_embeddings:
        kwargs[WORD_EMBEDDINGS_WEIGHT] = model.embedding.word_embeddings_weight
    else:
        kwargs[OUTPUT_WEIGHTS] = model.model_head.output_weights
    losses = {LanguageModelLossNames.multi_token_prediction_loss(i): [] for i in range(model._config.prediction_heads)}
    _ = model(input_, kwargs, losses=losses)
    for loss_name, loss_values in losses.items():
        Assert.gt(len(loss_values), 0)
    loss = sum(
        [
            sum(losses[LanguageModelLossNames.multi_token_prediction_loss(i)])
            for i in range(model._config.prediction_heads)
        ]
    )
    loss.backward()


@requires_cuda
@pytest.mark.skipif(not run_hybrid_test, reason="No CUDA available or Mamba not installed")
@pytest.mark.parametrize(
    ("hybrid_block_layout", "mtp_heads"),
    [
        (["m", "t"], ["t"]),
        (["t", "m2"], ["m2", "m"]),
        (["t", "m"], ["m", "t", "m2"]),
    ],
)
def test_hybrid_model_mtp(distributed_config, hybrid_block_layout, mtp_heads):
    hybrid_config = get_hybrid_config(hybrid_block_layout=hybrid_block_layout, mtp_heads=mtp_heads)
    model = HybridSSMBaseModel(hybrid_config, distributed_config)
    distributed = Distributed(distributed_config)
    model.setup(distributed)
    tensor_space = model._tensor_space
    materialize_meta_tensors(model, tensor_space)
    model.to("cuda")

    batch_size = 2
    seq_length = 32
    x = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
    position_ids = torch.arange(seq_length, device="cuda", dtype=torch.int64)
    attention_mask = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.bool)  # will be broadcasted to right shape
    labels = torch.randint(0, 49152, (batch_size, seq_length + model._config.prediction_heads - 1), device="cuda")
    losses = {
        LanguageModelLossNames.multi_token_prediction_loss(i): [] for i in range(len(model._config.mtp_heads) + 1)
    }
    kwargs = {
        "position_ids": position_ids,
        TransformerKwargs.sequence_first: False,
        TransformerKwargs.attention_mask: attention_mask,
        TransformerKwargs.attention_mask_value: -100,
        TransformerKwargs.grad_output: True,
        LanguageModelKwargs.labels: labels,
    }

    if model._config.tie_word_embeddings:
        kwargs[WORD_EMBEDDINGS_WEIGHT] = model.embedding.word_embeddings_weight
    else:
        kwargs[OUTPUT_WEIGHTS] = model.model_head.output_weights

    output = model(
        x,
        kwargs,
        losses=losses,
    )
    loss = sum(
        [
            sum(losses[LanguageModelLossNames.multi_token_prediction_loss(i)])
            for i in range(len(model._config.mtp_heads) + 1)
        ]
    )
    loss.backward()
