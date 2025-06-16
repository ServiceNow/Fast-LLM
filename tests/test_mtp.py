import typing

import pytest
import torch

from fast_llm.config import UpdateType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.language_model.embedding import WORD_EMBEDDINGS_WEIGHT
from fast_llm.layers.language_model.head import OUTPUT_WEIGHTS, LanguageModelHead
from fast_llm.layers.ssm.config import SSMBlockType
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.transformer.transformer import TransformerLayer
from fast_llm.models.gpt.config import GPTBaseModelConfig
from fast_llm.models.gpt.model import GPTBaseModel
from fast_llm.utils import Assert
from tests.common import get_hybrid_config, materialize_meta_tensors, requires_cuda

try:
    from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2
    from fast_llm.layers.ssm.mamba_layer import MambaLayer
    from fast_llm.models.ssm.model import HybridSSMBaseModel
except Exception:
    MambaLayer, HybridSSMBaseModel, DiscreteMamba2 = (
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


@pytest.mark.skip(reason="Too slow")
@requires_cuda
@pytest.mark.skipif(not run_hybrid_test, reason="No CUDA available or Mamba not installed")
@pytest.mark.parametrize(
    ("hybrid_block_layout", "prediction_heads", "default_mtp_type"),
    [
        ([SSMBlockType.mamba.value, SSMBlockType.transformer.value], 1, None),
        ([SSMBlockType.transformer.value, SSMBlockType.mamba.value], 2, None),
        ([SSMBlockType.mamba.value, SSMBlockType.transformer.value], 2, None),
        ([SSMBlockType.transformer.value, SSMBlockType.mamba2_discrete.value], 3, None),
        ([SSMBlockType.transformer.value, SSMBlockType.mamba2_discrete.value], 3, SSMBlockType.mamba.value),
    ],
)
def test_hybrid_model_mtp(distributed_config, hybrid_block_layout, prediction_heads, default_mtp_type):
    hybrid_config = get_hybrid_config(
        hybrid_block_layout=hybrid_block_layout, prediction_heads=prediction_heads, default_mtp_type=default_mtp_type
    )
    model = HybridSSMBaseModel(hybrid_config, distributed_config)
    distributed = Distributed(distributed_config)
    model.setup(distributed)
    tensor_space = model._tensor_space
    materialize_meta_tensors(model, tensor_space)
    model.to("cuda")

    num_heads, num_mtp_blocks = 0, 0
    str_block_mapping = {
        SSMBlockType.transformer: TransformerLayer,
        SSMBlockType.mamba: MambaLayer,
        SSMBlockType.mamba2_discrete: DiscreteMamba2,
    }
    mtp_block_type = default_mtp_type or hybrid_block_layout[-1]
    for block in model.get_output_layers():
        if isinstance(block, LanguageModelHead):
            num_heads += 1
        else:
            block = getattr(block, "mixer", block)
            Assert.custom(
                lambda _: isinstance(block, str_block_mapping[mtp_block_type]),
                f"Block {block} is not of type {str_block_mapping[mtp_block_type]}",
            )
            num_mtp_blocks += 1
    Assert.eq(num_heads, prediction_heads)
    Assert.eq(num_mtp_blocks, prediction_heads - 1)

    batch_size = 2
    seq_length = 32
    x = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
    position_ids = torch.arange(seq_length, device="cuda", dtype=torch.int64)
    attention_mask = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.bool)  # will be broadcasted to right shape
    labels = torch.randint(0, 49152, (batch_size, seq_length + model._config.prediction_heads - 1), device="cuda")
    losses = {LanguageModelLossNames.multi_token_prediction_loss(i): [] for i in range(model._config.prediction_heads)}
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
            for i in range(model._config.prediction_heads)
        ]
    )
    loss.backward()
