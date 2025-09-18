import typing

from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import MoEMLPConfig
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.engine.distributed.config import DistributedConfig
    from fast_llm.engine.distributed.distributed import Distributed
    from fast_llm.tensor import ParameterMeta


def get_init_megatron(
    meta: "ParameterMeta", config: DecoderBlockConfig, hidden_size: int
) -> typing.Callable[["torch.Tensor", "Distributed"], None]:
    def init_megatron(tensor: "torch.Tensor", distributed: "Distributed") -> None:
        Assert.eq(distributed.config.world_size, 1)
        if "bias" in meta.tensor_name:
            # Generator unused.
            return meta.param_init_method(meta, tensor, distributed.tp_init_generator)
        if "query" in meta.tensor_name or "key_value" in meta.tensor_name or "dense" in meta.tensor_name:
            tensor_ = _init_attention_megatron(config, meta, tensor, distributed, hidden_size)
        elif "position_embeddings" in meta.tensor_name:
            tensor_ = _init_position_embeddings_megatron(meta, tensor, distributed)
        elif "mlp.router.weight" in meta.tensor_name:
            tensor_ = _init_moe_router_megatron(meta, tensor, distributed)
        elif isinstance(config.mlp, MoEMLPConfig) and config.mlp.experts > 1 and "mlp.layer_" in meta.tensor_name:
            tensor_ = _init_moe_mlp_megatron(config, meta, tensor, distributed, hidden_size)
        elif "mlp.layer_2" in meta.tensor_name:
            tensor_ = _init_transposed_mlp_weight_megatron(meta, tensor, distributed)
        else:
            # Word embedding (override generator), layer norm (generator unused), other mlp weights.
            return meta.param_init_method(meta, tensor, distributed.tp_init_generator)
        tensor.copy_(tensor_.reshape_as(tensor))

    return init_megatron


def set_megatron_distributed_seeds(config: "DistributedConfig") -> None:
    # Shifts are hard-coded in Megatron.
    # Note: Megatron doesn't separate init generators so post-init random (dropout) won't match.
    config.dp_seed_shift = 0
    config.pp_seed_shift = 100
    config.pp_gen_init_seed_shift = 0
    config.tp_seed_shift = 1
    config.tp_gen_init_seed_shift = 2718
    config.reproducible_init = False


def _init_attention_megatron(
    config: DecoderBlockConfig,
    meta: "ParameterMeta",
    tensor: "torch.Tensor",
    distributed: "Distributed",
    hidden_size: int,
) -> "torch.Tensor":
    # Megatron combines q and kv and inverts the initialization order of qkv and dense layers.
    # It also always treats the tensors as tensor-parallel and uses a different rotary embedding format.
    assert meta.param_init_method is not None
    generator = distributed.tp_init_generator
    state = generator.get_state()
    # Initialize a mock dense layer to advance the random state
    meta.param_init_method(
        meta,
        dense_tensor_ := tensor.new_empty(
            config.mixer.head_size * config.mixer.heads,
            hidden_size,
        ),
        generator,
    )
    #  QKV is split differently. (Assuming no tensor-parallel.)
    heads_per_group = div(config.mixer.heads, config.mixer.head_groups)
    meta.param_init_method(
        meta,
        qkv_tensor_ := tensor.new_empty(
            config.mixer.head_groups,
            heads_per_group + 2,
            config.mixer.head_size,
            hidden_size,
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

    if isinstance(config.mixer.rotary, DefaultRotaryConfig) and config.mixer.rotary.complex_format:
        from fast_llm.layers.attention.rotary.config import convert_rotary_real_to_complex

        # Megatron uses (2, head_size/2) for the complex split; we use (head_size/2, 2).
        # TODO: Avoid unnecessarily changing the value and dense tensors.
        tensor_ = convert_rotary_real_to_complex(tensor_.view_as(meta), config.mixer.head_size, kv_dim)
    return tensor_


def _init_position_embeddings_megatron(
    meta: "ParameterMeta", tensor: "torch.Tensor", distributed: "Distributed"
) -> "torch.Tensor":
    import torch

    # Megatron initializes the position embeddings on cpu twice.
    assert meta.param_init_method is not None
    generator = distributed.default_cpu_generator
    meta.param_init_method(meta, tensor_ := torch.empty(tensor.shape, dtype=tensor.dtype), generator)
    meta.param_init_method(meta, tensor_, generator)
    return tensor_


def _init_transposed_mlp_weight_megatron(
    meta: "ParameterMeta", tensor: "torch.Tensor", distributed: "Distributed"
) -> "torch.Tensor":
    import torch

    # Megatron never transposes the mlp layer 2 weight.
    assert meta.param_init_method is not None
    meta.param_init_method(meta, tensor_ := torch.empty_like(tensor), distributed.tp_init_generator)
    return tensor_.view(meta.size(1), meta.size(0)).t()


def _init_moe_router_megatron(
    meta: "ParameterMeta", tensor: "torch.Tensor", distributed: "Distributed"
) -> "torch.Tensor":
    import torch

    # Megatron initializes the router on cpu.
    assert meta.param_init_method is not None
    meta.param_init_method(
        meta, tensor_ := torch.empty(tensor.shape, dtype=tensor.dtype), distributed.default_cpu_generator
    )
    return tensor_


def _init_moe_mlp_megatron(
    config: DecoderBlockConfig,
    meta: "ParameterMeta",
    tensor: "torch.Tensor",
    distributed: "Distributed",
    hidden_size: int,
) -> "torch.Tensor":
    assert meta.param_init_method is not None
    generator = distributed.tp_init_generator if meta.is_tensor_parallel else distributed.pp_init_generator
    # self.param_init_method(self, tensor, generator)
    state = generator.get_state()
    weight_1 = tensor.new_empty(
        config.mlp.experts * (1 + config.mlp.gated) * config.mlp.intermediate_size, hidden_size
    )
    weight_2 = tensor.new_empty(config.mlp.experts * config.mlp.intermediate_size, hidden_size)
    for chunk_1, chunk_2 in zip(weight_1.chunk(config.mlp.experts), weight_2.chunk(config.mlp.experts)):
        meta.param_init_method(meta, chunk_1, generator)
        chunk_2_ = chunk_2.new_empty(hidden_size, config.mlp.intermediate_size)
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
