import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import gather_op, reduce_op, reduce_scatter_op, swap_mult_dim
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.rotary import apply_rotary_embeddings
from fast_llm.functional.triton.rotary import triton_rotary_autograd_
from fast_llm.layers.common.linear import InputParallelLinear, OutputParallelLinear
from fast_llm.layers.transformer.config import (
    TransformerConfig,
    TransformerDimNames,
    TransformerKwargs,
    TransformerSubLayerName,
)
from fast_llm.logging import log_distributed_grad, log_distributed_tensor
from fast_llm.tensor import TensorMeta, init_normal_, init_zeros_
from fast_llm.utils import Assert

try:
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # noqa
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func

    _flash_available = True
except ImportError:
    _flash_available = False


class AttachGrad(torch.autograd.Function):
    """
    "Attach" the gradient of y to that of x,
    so that the gradient of y is automatically added to that of x during the gradient computation of x.
    The gradient of y should be computed first.

    In practice this allows inserting a breaking point in autograd to
    split the gradient computation of x in two separate backward calls,
    by setting `y = x.detach().requires_grad_()`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa
        # TODO: can we do it without saving y? (We only need its grad)
        ctx.save_for_backward(y)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # noqa
        (y,) = ctx.saved_tensors
        grad = y.grad + grad_output
        return grad, None


class Attention(torch.nn.Module):
    """
    A self-attention layer.
    """

    _QUERY_DIMS = (
        TransformerDimNames.batch,
        TransformerDimNames.sequence_q,
        TransformerDimNames.composite_heads,
        TransformerDimNames.kv_channels,
    )
    _KV_DIMS = (
        TransformerDimNames.batch,
        TransformerDimNames.sequence_q,
        TransformerDimNames.group_heads,
        TransformerDimNames.kv_channels,
    )
    _CONTEXT_DIMS = (
        TransformerDimNames.batch,
        TransformerDimNames.sequence_q,
        TransformerDimNames.composite_dense,
    )

    def __init__(
        self,
        config: TransformerConfig,
        tensor_space: TensorSpace,
        layer_index,
    ):
        super().__init__()
        self._config = config
        self._tensor_space = tensor_space
        Assert.in_range_incl(layer_index, 1, self._config.num_layers)
        self._layer_index = layer_index
        self._sequence_parallel = self._tensor_space.distributed_config.sequence_tensor_parallel
        self._debug_transformer = self._config.debug_transformer
        self._use_flash_attention = self._config.do_use_flash_attention(self._tensor_space.distributed_config)

        init_method_qkv = init_normal_(
            std=self._config.init_method_std_qkv,
            min_val=self._config.init_method_min_qkv,
            max_val=self._config.init_method_max_qkv,
        )
        init_method_std_attn_proj = init_normal_(
            std=self._config.init_method_std_attn_proj,
            min_val=self._config.init_method_min_attn_proj,
            max_val=self._config.init_method_max_attn_proj,
        )

        self._kv_channels = self._tensor_space.get_tensor_dim(TransformerDimNames.kv_channels).size
        self._head_groups = self._tensor_space.get_tensor_dim(TransformerDimNames.head_groups).global_size
        self._local_head_groups = self._tensor_space.get_tensor_dim(TransformerDimNames.head_groups).size
        self._local_heads_per_group = self._tensor_space.get_tensor_dim(TransformerDimNames.group_heads).size
        self._local_heads = self._local_head_groups * self._local_heads_per_group
        self._softmax_scale = self._kv_channels ** (-self._config.attention_softmax_scale_power)

        hidden_dim = self._tensor_space.get_tensor_dim(TransformerDimNames.hidden)

        # TODO: Merge the query and key-value computations? (harder with sequence parallel.)
        self.query = OutputParallelLinear(
            hidden_dim,
            self._tensor_space.get_tensor_dim(TransformerDimNames.composite_query),
            bias=self._config.add_attn_qkv_bias,
            weight_init_method=init_method_qkv,
            bias_init_method=init_method_qkv if self._config.random_bias_init else init_zeros_,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._config.attention_lr_scale,
        )
        self.key_value = OutputParallelLinear(
            hidden_dim,
            self._tensor_space.get_tensor_dim(TransformerDimNames.composite_key_value),
            bias=self._config.add_attn_qkv_bias,
            weight_init_method=init_method_qkv,
            bias_init_method=init_method_qkv if self._config.random_bias_init else init_zeros_,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._config.attention_lr_scale,
        )
        self._query_key_value = wrap_forward_backward(self._query_key_value_forward, self._query_key_value_backward)

        # Output.
        self.dense = InputParallelLinear(
            self._tensor_space.get_tensor_dim(TransformerDimNames.composite_dense),
            hidden_dim,
            bias=self._config.add_attn_dense_bias,
            weight_init_method=init_method_std_attn_proj,
            bias_init_method=init_method_std_attn_proj if self._config.random_bias_init else init_zeros_,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._config.attention_lr_scale,
        )

        # PEFT.
        self.query = self._config.peft.apply_linear(self.query, TransformerSubLayerName.query)
        self.key_value = self._config.peft.apply_linear(self.key_value, TransformerSubLayerName.key_value)
        self.dense = self._config.peft.apply_linear(self.dense, TransformerSubLayerName.dense)

    def _attn_fused(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor
    ) -> torch.Tensor:
        # Backup attention (inefficient)
        b, sq, hidden = query.shape
        sk = key.size(1)

        if self._local_head_groups == 1:
            query = query.view(b, sq * self._local_heads, self._kv_channels)
            key = key.transpose(-1, -2)
        else:
            query = (
                query.unflatten(-1, (self._local_head_groups, self._local_heads_per_group, self._kv_channels))
                .transpose(1, 2)
                .reshape(b * self._local_head_groups, sq * self._local_heads_per_group, self._kv_channels)
            )
            key = key.unflatten(-1, (self._local_head_groups, self._kv_channels)).movedim(1, 3).flatten(0, 1)
            value = value.unflatten(-1, (self._local_head_groups, self._kv_channels)).transpose(1, 2).flatten(0, 1)

        attn_weights = torch.empty(
            (b * self._local_head_groups, sq * self._local_heads_per_group, sk), device=query.device, dtype=query.dtype
        )
        attn_weights = torch.baddbmm(
            attn_weights,
            query,
            key,
            beta=0,
            alpha=self._softmax_scale / self._layer_index,
        ).view(b, self._local_head_groups, sq, self._local_heads_per_group, sk)

        attn_weights = attn_weights.to(torch.float32) * self._layer_index
        attn_weights = torch.where(mask, attn_weights, mask_value)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(query.dtype)

        with set_generator(self._tensor_space.distributed.tp_generator):
            attn_weights = torch.dropout(attn_weights, self._config.attention_dropout, self.training)
        attn_output = torch.bmm(
            attn_weights.view(b * self._local_head_groups, sq * self._local_heads_per_group, sk), value
        )

        if self._local_head_groups == 1:
            return attn_output.view(b, sq, -1)
        else:
            return (
                attn_output.view(b, self._local_head_groups, sq, self._local_heads_per_group, self._kv_channels)
                .transpose(1, 2)
                .flatten(2)
            )

    def _get_meta(
        self, input_: torch.Tensor, name: str, dim_names: tuple[str, ...], kwargs: dict[str, typing.Any]
    ) -> TensorMeta:
        hidden_dims = {dim.name: dim for dim in kwargs[TransformerKwargs.hidden_dims]}
        return TensorMeta.from_dims(
            tuple(
                hidden_dims[dim_name] if dim_name in hidden_dims else self._tensor_space.get_tensor_dim(dim_name)
                for dim_name in dim_names
            ),
            tensor_name=f"transformer layer {self._layer_index} attn {name}",
            dtype=input_.dtype,
        )

    def _debug_log(
        self, tensor: torch.Tensor, name: str, dim_names: tuple[str, ...], kwargs: dict[str, typing.Any]
    ) -> None:
        # TODO: Local vs global
        Assert.gt(self._debug_transformer, 0)
        log_distributed_tensor(
            "",
            tensor,
            level=self._debug_transformer,
            meta=self._get_meta(tensor, name, dim_names, kwargs),
            distributed=self._tensor_space.distributed,
        )
        if tensor.requires_grad:
            log_distributed_grad(
                "",
                tensor,
                level=self._debug_transformer,
                meta=self._get_meta(tensor, name + " grad", dim_names, kwargs),
                distributed=self._tensor_space.distributed,
            )

    def _query_key_value_forward(
        self, input_: torch.Tensor, sequence_first: bool
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, typing.Any]]:
        key_value, key_value_context = self.key_value.forward_only(input_)

        handle = None

        if self._head_groups == 1 and self._sequence_parallel:
            key_value, handle = gather_op(
                key_value, group=self._tensor_space.distributed.tensor_group, dim=0, async_op=True
            )

        if self._tensor_space.distributed.sequence_data_group:
            if handle:
                # TODO: This is probably unnecessary.
                handle.wait()
            # sequence dim may not be zero, but this needs to be handled after `handle.wait()`
            key_value, handle = gather_op(
                key_value, group=self._tensor_space.distributed.sequence_data_group, dim=0, async_op=True
            )

        query, query_context = self.query.forward_only(input_)

        if handle:
            handle.wait()

        if self._tensor_space.distributed.sequence_data_group and not sequence_first:
            key_value = swap_mult_dim(key_value, self._tensor_space.distributed_config.sequence_data_parallel, 0, 1)

        context = {"query": query_context, "key_value": key_value_context, "sequence_first": sequence_first}
        return query, key_value, context

    def _query_key_value_backward(
        self, query_grad: torch.Tensor, key_value_grad: torch.Tensor, context: dict
    ) -> torch.Tensor:
        # TODO: De-allocate qkv grads quicker.
        handle = None

        if self._tensor_space.distributed.sequence_data_group:
            key_value_grad, handle = reduce_scatter_op(
                key_value_grad,
                group=self._tensor_space.distributed.sequence_data_group,
                dim=1 - context["sequence_first"],
                async_op=True,
            )

        # TODO: Overlap with both.
        input_grad = self.query.backward(query_grad, context.pop("query"))

        if handle:
            handle.wait()

        if self._head_groups == 1 and (group := self._tensor_space.distributed.tensor_group):
            if self._sequence_parallel:
                key_value_grad = reduce_scatter_op(key_value_grad, group=group, dim=0)
            else:
                key_value_grad = reduce_op(key_value_grad, group=group)

        input_grad.add_(self.key_value.backward(key_value_grad, context.pop("key_value")))
        return input_grad

    def _decide_window_size(self) -> int | None:
        # NOTE: This is a temporal solution for qwen 2.X
        # https://github.com/huggingface/transformers/blob/5e2183f344911aa82aba0b83778a4f196cff378e/src/transformers/models/qwen2/modular_qwen2.py#L71
        # TODO: make universal per layer config
        window_size = self._config.window_size
        if self._config.max_window_layers is not None and self._layer_index < self._config.max_window_layers:
            window_size = None

        return window_size

    def forward(self, input_: torch.Tensor, kwargs: dict[str, typing.Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the attention layer.
        """
        sequence_first = kwargs[TransformerKwargs.sequence_first]
        query, key_value, context = self._query_key_value(input_, sequence_first)

        # Get the attention mask
        mask = kwargs.get(TransformerKwargs.attention_mask)
        mask_value = kwargs.get(TransformerKwargs.attention_mask_value, -float("inf"))

        # Create causal mask if not using bidirectional attention
        if not self._config.bidirectional_attention and mask is None:
            sequence_q = query.size(1)
            sequence_k = key_value.size(1)
            # Create causal mask [sequence_q, sequence_k]
            causal_mask = torch.triu(
                torch.ones(sequence_q, sequence_k, dtype=torch.bool, device=query.device),
                diagonal=1,
            )
            mask = ~causal_mask
            mask_value = -float("inf")

        # Apply rotary embeddings if enabled
        if self._config.rotary.enabled:
            if self._config.rotary.triton:
                triton_rotary_autograd_(
                    query,
                    kwargs[TransformerKwargs.rotary_freq_q],
                    sequence_first=sequence_first,
                )
                triton_rotary_autograd_(
                    key_value,
                    kwargs[TransformerKwargs.rotary_freq_k],
                    sequence_first=sequence_first,
                )
            else:
                query = apply_rotary_embeddings(
                    query,
                    kwargs[TransformerKwargs.rotary_freq_q],
                    sequence_first=sequence_first,
                )
                key_value = apply_rotary_embeddings(
                    key_value,
                    kwargs[TransformerKwargs.rotary_freq_k],
                    sequence_first=sequence_first,
                )

        # Split key and value
        key, value = key_value.chunk(2, dim=-1)

        # Compute attention
        if self._use_flash_attention:
            # Flash attention
            sequence_lengths = kwargs.get(TransformerKwargs.sequence_lengths)
            if sequence_lengths is not None:
                # Variable length sequences
                cu_seqlens_q = kwargs[TransformerKwargs.cu_seqlens_q]
                cu_seqlens_k = kwargs[TransformerKwargs.cu_seqlens_k]
                max_seqlen_q = kwargs[TransformerKwargs.max_seqlen_q]
                max_seqlen_k = kwargs[TransformerKwargs.max_seqlen_k]
                attn_output = _flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self._softmax_scale / self._layer_index,
                    dropout_p=self._config.attention_dropout if self.training else 0.0,
                    causal=not self._config.bidirectional_attention,
                    return_attn_probs=False,
                )
            else:
                # Fixed length sequences
                attn_output = _flash_attn_func(
                    query,
                    key,
                    value,
                    self._softmax_scale / self._layer_index,
                    dropout_p=self._config.attention_dropout if self.training else 0.0,
                    causal=not self._config.bidirectional_attention,
                    return_attn_probs=False,
                )
        else:
            # Regular attention
            attn_output = self._attn_fused(query, key, value, mask, mask_value)

        # Project output
        output = self.dense(attn_output)

        if self._debug_transformer:
            self._debug_log(output, "dense", self._CONTEXT_DIMS, kwargs)

        return output, None
