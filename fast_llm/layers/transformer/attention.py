import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import gather_op, reduce_op, reduce_scatter_op, swap_mult_dim
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.layers.common.linear import InputParallelLinear, OutputParallelLinear
from fast_llm.layers.transformer.config import (
    TransformerConfig,
    TransformerDimNames,
    TransformerKwargs,
    TransformerSubLayerName,
)
from fast_llm.logging import log_distributed_grad, log_distributed_tensor
from fast_llm.tensor import TensorMeta, init_normal_, init_zeros_
from fast_llm.utils import Assert, get_lr_scale

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
        # Assert.in_range_incl(layer_index, 1, max(self._config.num_layers, 1))
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

        layer_lr_scale = config.per_layer_lr_scale[layer_index] if config.per_layer_lr_scale else None
        attention_lr_scale = get_lr_scale(self._config.attention_lr_scale, layer_lr_scale)

        # TODO: Merge the query and key-value computations? (harder with sequence parallel.)
        self.query = OutputParallelLinear(
            hidden_dim,
            self._tensor_space.get_tensor_dim(TransformerDimNames.composite_query),
            bias=self._config.add_attn_qkv_bias,
            weight_init_method=init_method_qkv,
            bias_init_method=init_method_qkv if self._config.random_bias_init else init_zeros_,
            sequence_parallel=self._sequence_parallel,
            lr_scale=attention_lr_scale,
        )
        self.key_value = OutputParallelLinear(
            hidden_dim,
            self._tensor_space.get_tensor_dim(TransformerDimNames.composite_key_value),
            bias=self._config.add_attn_qkv_bias,
            weight_init_method=init_method_qkv,
            bias_init_method=init_method_qkv if self._config.random_bias_init else init_zeros_,
            sequence_parallel=self._sequence_parallel,
            lr_scale=attention_lr_scale,
        )
        self._query_key_value = wrap_forward_backward(self._query_key_value_forward, self._query_key_value_backward)

        # Rotary embeddings.
        self._rotary = self._config.rotary.build()

        # Output.
        self.dense = InputParallelLinear(
            self._tensor_space.get_tensor_dim(TransformerDimNames.composite_dense),
            hidden_dim,
            bias=self._config.add_attn_dense_bias,
            weight_init_method=init_method_std_attn_proj,
            bias_init_method=init_method_std_attn_proj if self._config.random_bias_init else init_zeros_,
            sequence_parallel=self._sequence_parallel,
            lr_scale=attention_lr_scale,
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
        sequence_first = kwargs[TransformerKwargs.sequence_first]
        query, key_value = self._query_key_value(input_, sequence_first)

        # TODO: Move the rest to function.

        if (past_key_values := kwargs.get(TransformerKwargs.past_key_values)) is not None:
            assert sequence_first
            # Clear the lists so tensors can be de-allocated
            key_value = torch.cat((past_key_values.pop(0), key_value), dim=0)

        if (presents := kwargs.get(TransformerKwargs.presents)) is not None:
            # Return the presents as a leaf tensors so the gradients from later micro-sequences
            # don't propagate to this one.
            presents.append(present := key_value.detach().requires_grad_())
            # Manually add the gradients from later micro-sequences.
            key_value = AttachGrad.apply(key_value, present)

        if self._tensor_space.distributed.sequence_data_group:
            key_value = (
                key_value[: kwargs[TransformerKwargs.sequence_k_dim].size]
                if sequence_first
                else key_value[:, : kwargs[TransformerKwargs.sequence_k_dim].size]
            )

        if sequence_first:
            # TODO: Optimize (is contiguous avoidable?)
            query = query.transpose(0, 1).contiguous()
            key_value = key_value.transpose(0, 1).contiguous()

        key, value = key_value.split(self._local_head_groups * self._kv_channels, dim=-1)

        query = query.view(*query.shape[:2], self._local_heads, self._kv_channels)
        key = key.view(*key.shape[:2], self._local_head_groups, self._kv_channels)
        value = value.view(*value.shape[:2], self._local_head_groups, self._kv_channels)

        if self._debug_transformer:
            self._debug_log(query, "query_rotary_input", self._QUERY_DIMS, kwargs)
            self._debug_log(
                key,
                "key_rotary_input",
                self._KV_DIMS,
                kwargs,
            )
        query, key = self._rotary(query, key, kwargs)

        window_size = self._decide_window_size()

        if self._use_flash_attention:
            assert _flash_available
            with set_generator(self._tensor_space.distributed.tp_generator):
                if (cu_seqlens_q := kwargs.get(TransformerKwargs.cu_seqlens_q, None)) is not None:
                    out_dims = query.size()
                    query = query.view(-1, query.size(-2), query.size(-1))
                    key = key.view(-1, key.size(-2), key.size(-1))
                    value = value.view(-1, value.size(-2), value.size(-1))
                    input_ = _flash_attn_varlen_func(
                        query,
                        key,
                        value,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=kwargs.get(TransformerKwargs.cu_seqlens_k),
                        max_seqlen_q=kwargs.get(TransformerKwargs.max_seqlen_q),
                        max_seqlen_k=kwargs.get(TransformerKwargs.max_seqlen_k),
                        dropout_p=self._config.attention_dropout if self.training else 0.0,
                        window_size=(-1, -1) if window_size is None else (window_size - 1, 0),
                        causal=True,
                        softmax_scale=self._softmax_scale,
                    ).view(*out_dims)
                else:
                    input_ = _flash_attn_func(
                        query,
                        key,
                        value,
                        window_size=(-1, -1) if window_size is None else (window_size - 1, 0),
                        dropout_p=self._config.attention_dropout if self.training else 0.0,
                        causal=True,
                        softmax_scale=self._softmax_scale,
                    )
            input_ = input_.flatten(-2)
        else:
            # TODO: Avoid the flattens.
            input_ = self._attn_fused(
                query.flatten(-2),
                key.flatten(-2),
                value.flatten(-2),
                kwargs[TransformerKwargs.attention_mask],
                kwargs[TransformerKwargs.attention_mask_value],
            )

        if self._debug_transformer:
            self._debug_log(query, "query", self._QUERY_DIMS, kwargs)
            self._debug_log(
                key,
                "key",
                self._KV_DIMS,
                kwargs,
            )
            self._debug_log(
                value,
                "value",
                self._KV_DIMS,
                kwargs,
            )
            self._debug_log(input_, "context", self._CONTEXT_DIMS, kwargs)

        if sequence_first:
            # TODO: Optimize (is contiguous avoidable? Transpose dense output?)
            input_ = input_.transpose(0, 1).contiguous()
        return self.dense(input_)
