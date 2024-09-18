import torch

from fast_llm.distributed import Distributed, DistributedConfig
from fast_llm.functional.rotary import convert_rotary_real_to_complex
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.tensor import ParameterMeta
from fast_llm.utils import Assert, div


def get_init_megatron(meta: ParameterMeta, config: TransformerConfig):
    def init_megatron(tensor: torch.Tensor, distributed: Distributed):
        Assert.eq(distributed.config.world_size, 1)
        if "bias" in meta.tensor_name:
            # Generator unused.
            return meta.param_init_method(meta, tensor, distributed.tp_init_generator)
        if "query" in meta.tensor_name or "key_value" in meta.tensor_name or "dense" in meta.tensor_name:
            tensor_ = _init_attention_megatron(config, meta, tensor, distributed)
        elif "position_embeddings" in meta.tensor_name:
            tensor_ = _init_position_embeddings_megatron(meta, tensor, distributed)
        elif "mlp.router.weight" in meta.tensor_name:
            tensor_ = _init_moe_router_megatron(meta, tensor, distributed)
        elif config.num_experts > 1 and "mlp.layer_" in meta.tensor_name:
            tensor_ = _init_moe_mlp_megatron(config, meta, tensor, distributed)
        elif "mlp.layer_2" in meta.tensor_name:
            tensor_ = _init_transposed_mlp_weight_megatron(config, meta, tensor, distributed)
        else:
            # Word embedding (override generator), layer norm (generator unused), other mlp weights.
            return meta.param_init_method(meta, tensor, distributed.tp_init_generator)
        return tensor.copy_(tensor_.reshape_as(tensor))

    return init_megatron


def set_megatron_distributed_seeds(config: DistributedConfig):
    # Only single-gpu is supported.
    Assert.eq(config.world_size, 1)
    # Shifts are hard-coded in Megatron.
    # Note: Megatron doesn't separate init generators so post-init random (dropout) won't match.
    config.dp_seed_shift = 0
    config.pp_seed_shift = 100
    config.pp_gen_init_seed_shift = 0
    config.tp_seed_shift = 1
    config.tp_gen_init_seed_shift = 2718
    config.reproducible_init = False


def _init_attention_megatron(
    config: TransformerConfig, meta: ParameterMeta, tensor: torch.Tensor, distributed: Distributed
):
    # Megatron combines q and kv and inverts the initialization order of qkv and dense layers.
    # It also always treats the tensors as tensor-parallel and uses a different rotary embedding format.
    assert meta.param_init_method is not None
    generator = distributed.tp_init_generator
    state = generator.get_state()
    # Initialize a mock dense layer to advance the random state
    dense_tensor_ = meta.param_init_method(
        meta,
        tensor.new_empty(
            config.kv_channels * config.num_attention_heads,
            config.hidden_size,
        ),
        generator,
    )
    #  QKV is split differently. (Assuming no tensor-parallel.)
    heads_per_group = div(config.num_attention_heads, config.head_groups)
    qkv_tensor_ = meta.param_init_method(
        meta,
        tensor.new_empty(
            config.head_groups,
            heads_per_group + 2,
            config.kv_channels,
            config.hidden_size,
        ),
        generator,
    )
    if "dense" in meta.tensor_name:
        kv_dim = 1
        tensor_ = dense_tensor_
    else:
        # Keep the original random state for key_value and dense.
        generator.set_state(state)
        kv_dim = 0
        if "query" in meta.tensor_name:
            # We want to generate the same tensor for key_value.
            tensor_ = qkv_tensor_[:, :heads_per_group]
        elif "key_value" in meta.tensor_name:
            tensor_ = qkv_tensor_[:, heads_per_group:].transpose(0, 1)
        else:
            raise NotImplementedError(meta.tensor_name)

    if config.use_rotary_position_embeddings and config.complex_rotary_embeddings:
        # Megatron uses (2, kv_channels/2) for the complex split; we use (kv_channels/2, 2).
        # TODO: Avoid unnecessarily changing the value and dense tensors.
        tensor_ = convert_rotary_real_to_complex(tensor_.view_as(meta), config.kv_channels, kv_dim)
    return tensor_


def _init_position_embeddings_megatron(meta: ParameterMeta, tensor: torch.Tensor, distributed: Distributed):
    # Megatron initializes the position embeddings on cpu twice.
    assert meta.param_init_method is not None
    generator = distributed.default_cpu_generator
    tensor_ = meta.param_init_method(meta, torch.empty(tensor.shape, dtype=tensor.dtype), generator)
    return meta.param_init_method(meta, tensor_, generator)


def _init_transposed_mlp_weight_megatron(
    config: TransformerConfig, meta: ParameterMeta, tensor: torch.Tensor, distributed: Distributed
):
    # Megatron never transposes the mlp layer 2 weight.
    assert meta.param_init_method is not None
    tensor_ = meta.param_init_method(meta, torch.empty_like(tensor), distributed.tp_init_generator)
    if config.transposed_mlp_weight:
        tensor_ = tensor_.view(meta.size(1), meta.size(0)).t()
    return tensor_


def _init_moe_router_megatron(meta: ParameterMeta, tensor: torch.Tensor, distributed: Distributed):
    # Megatron initializes the router on cpu.
    assert meta.param_init_method is not None
    tensor_ = meta.param_init_method(
        meta, torch.empty(tensor.shape, dtype=tensor.dtype), distributed.default_cpu_generator
    )
    return tensor_


def _init_moe_mlp_megatron(
    config: TransformerConfig, meta: ParameterMeta, tensor: torch.Tensor, distributed: Distributed
):
    assert meta.param_init_method is not None
    generator = distributed.tp_init_generator if meta.is_tensor_parallel else distributed.pp_init_generator
    # self.param_init_method(self, tensor, generator)
    state = generator.get_state()
    weight_1 = tensor.new_empty(config.num_experts * (1 + config.gated) * config.ffn_hidden_size, config.hidden_size)
    weight_2 = tensor.new_empty(config.num_experts * config.ffn_hidden_size, config.hidden_size)
    for chunk_1, chunk_2 in zip(weight_1.chunk(config.num_experts), weight_2.chunk(config.num_experts)):
        meta.param_init_method(meta, chunk_1, generator)
        chunk_2_ = chunk_2.new_empty(config.hidden_size, config.ffn_hidden_size)
        meta.param_init_method(meta, chunk_2_, generator)
        chunk_2.copy_(chunk_2_.t())
    if "layer_1.weight" in meta.tensor_name:
        # Keep the original random state for weight_2.
        generator.set_state(state)
        tensor_ = weight_1
    elif "layer_2.weight" in meta.tensor_name:
        tensor_ = weight_2
    else:
        raise NotImplementedError(meta.tensor_name)

    return tensor_
