import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import gather_op, reduce_op, reduce_scatter_op, swap_mult_dim
from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.layers.attention.config import AttentionConfig, AttentionKwargs
from fast_llm.layers.block.config import BlockDimNames
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.tensor import TensorMeta
from fast_llm.utils import div

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


class Attention[ConfigType: AttentionConfig](BlockWithBias[ConfigType]):
    """
    A self-attention layer.
    """

    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )
        self._use_flash_attention = self._config.do_use_flash_attention(self._distributed_config)

        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        self._sequence_data_parallel_dim = self._distributed_config.get_distributed_dim(
            DistributedDimNames.sequence_data
        )
        head_group_dim = TensorDim(
            "head_groups", self._config.head_groups, self._parallel_dim if self._config.head_groups > 1 else None
        )
        group_heads_dim = TensorDim(
            "group_heads",
            div(self._config.heads, self._config.head_groups),
            None if self._config.head_groups > 1 else self._parallel_dim,
        )
        self._local_head_groups = head_group_dim.size
        self._local_heads_per_group = group_heads_dim.size
        self._local_heads = self._local_head_groups * self._local_heads_per_group

        head_size_dim = TensorDim("head_size", self._config.head_size)
        query_dim = CompositeTensorDim("query", (head_group_dim, group_heads_dim, head_size_dim))
        key_value_dim = ConcatenatedTensorDim(
            "key_value",
            (
                CompositeTensorDim("key", (head_group_dim, head_size_dim)),
                CompositeTensorDim("value", (head_group_dim, head_size_dim)),
            ),
        )
        dense_dim = CompositeTensorDim("dense", (head_group_dim, group_heads_dim, head_size_dim))

        self._softmax_scale = self._config.head_size ** (-self._config.softmax_scale_power)

        # TODO: Merge the query and key-value computations? (harder with sequence parallel.)
        self.query = self._config.query_layer.get_layer(
            hidden_dim,
            query_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=self._config.add_linear_biases,
            default_apply_peft=True,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        # TODO: Use value config.
        self.key_value = self._config.key_layer.get_layer(
            hidden_dim,
            key_value_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=self._config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=None if self._config.key_layer.apply_peft is None else self._peft,
        )
        if self._peft is not None and self._config.key_layer.apply_peft is None:
            # Default: Apply to value only.
            # TODO: Avoid this hack.
            self.key_value = self._peft.apply_linear(
                self.key_value, True, out_channel_begin=div(key_value_dim.global_size, 2)
            )

        self._query_key_value = wrap_forward_backward(self._query_key_value_forward, self._query_key_value_backward)

        # Rotary embeddings.
        self._rotary = self._config.rotary.get_layer(head_size_dim)

        # Output.
        self.dense = self._config.dense_layer.get_layer(
            dense_dim,
            hidden_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=self._config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        # Debug dims
        self._query_dims = (
            BlockDimNames.batch,
            BlockDimNames.sequence_q,
            CompositeTensorDim("heads", (head_group_dim, group_heads_dim)),
            head_size_dim,
        )
        self._kv_dims = (
            BlockDimNames.batch,
            BlockDimNames.sequence_q,
            head_group_dim,
            head_size_dim,
        )
        self._context_dims = (
            BlockDimNames.batch,
            BlockDimNames.sequence_q,
            dense_dim,
        )

    def _attn_fused(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor
    ) -> torch.Tensor:
        # Backup attention (inefficient)
        b, sq, hidden = query.shape
        sk = key.size(1)

        if self._local_head_groups == 1:
            query = query.view(b, sq * self._local_heads, self._config.head_size)
            key = key.transpose(-1, -2)
        else:
            query = (
                query.unflatten(-1, (self._local_head_groups, self._local_heads_per_group, self._config.head_size))
                .transpose(1, 2)
                .reshape(b * self._local_head_groups, sq * self._local_heads_per_group, self._config.head_size)
            )
            key = key.unflatten(-1, (self._local_head_groups, self._config.head_size)).movedim(1, 3).flatten(0, 1)
            value = (
                value.unflatten(-1, (self._local_head_groups, self._config.head_size)).transpose(1, 2).flatten(0, 1)
            )

        attn_weights = torch.empty(
            (b * self._local_head_groups, sq * self._local_heads_per_group, sk), device=query.device, dtype=query.dtype
        )
        attn_weights = torch.baddbmm(
            attn_weights,
            query,
            key,
            beta=0,
            alpha=self._softmax_scale,
        ).view(b, self._local_head_groups, sq, self._local_heads_per_group, sk)

        attn_weights = attn_weights.to(torch.float32)
        attn_weights = torch.where(mask, attn_weights, mask_value)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(query.dtype)

        with set_generator(self._distributed.tp_generator):
            attn_weights = torch.dropout(attn_weights, self._config.dropout, self.training)
        attn_output = torch.bmm(
            attn_weights.view(b * self._local_head_groups, sq * self._local_heads_per_group, sk), value
        )

        if self._local_head_groups == 1:
            return attn_output.view(b, sq, -1)
        else:
            return (
                attn_output.view(b, self._local_head_groups, sq, self._local_heads_per_group, self._config.head_size)
                .transpose(1, 2)
                .flatten(2)
            )

    def _query_key_value_forward(
        self, input_: torch.Tensor, sequence_first: bool
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, typing.Any]]:
        key_value, key_value_context = self.key_value.forward_only(input_)

        handle = None

        if self._config.head_groups == 1 and self._sequence_parallel:
            key_value, handle = gather_op(key_value, group=self._parallel_dim.group, dim=0, async_op=True)

        if self._sequence_data_parallel_dim.group:
            if handle:
                # TODO: This is probably unnecessary.
                handle.wait()
            # sequence dim may not be zero, but this needs to be handled after `handle.wait()`
            key_value, handle = gather_op(
                key_value, group=self._sequence_data_parallel_dim.group, dim=0, async_op=True
            )

        query, query_context = self.query.forward_only(input_)

        if handle:
            handle.wait()

        if self._sequence_data_parallel_dim.group and not sequence_first:
            key_value = swap_mult_dim(key_value, self._sequence_parallel, 0, 1)

        context = {"query": query_context, "key_value": key_value_context, "sequence_first": sequence_first}
        return query, key_value, context

    def _query_key_value_backward(
        self, query_grad: torch.Tensor, key_value_grad: torch.Tensor, context: dict
    ) -> torch.Tensor:
        # TODO: De-allocate qkv grads quicker.
        key_value_grad, handle = reduce_scatter_op(
            key_value_grad,
            group=self._sequence_data_parallel_dim.group,
            dim=1 - context["sequence_first"],
            async_op=True,
        )

        # TODO: Overlap with both.
        input_grad = self.query.backward(query_grad, context.pop("query"))

        if handle:
            handle.wait()

        if self._config.head_groups == 1 and (group := self._parallel_dim.group):
            if self._sequence_parallel:
                key_value_grad = reduce_scatter_op(key_value_grad, group=group, dim=0)
            else:
                key_value_grad = reduce_op(key_value_grad, group=group)

        input_grad.add_(self.key_value.backward(key_value_grad, context.pop("key_value")))
        return input_grad

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sequence_first = kwargs[AttentionKwargs.sequence_first]
        query, key_value = self._query_key_value(input_, sequence_first)

        # TODO: Move the rest to function.

        if (past_key_values := kwargs.get(AttentionKwargs.past_key_values)) is not None:
            assert sequence_first
            # Clear the lists so tensors can be de-allocated
            key_value = torch.cat((past_key_values.pop(0), key_value), dim=0)

        if (presents := kwargs.get(AttentionKwargs.presents)) is not None:
            # Return the presents as a leaf tensors so the gradients from later micro-sequences
            # don't propagate to this one.
            presents.append(present := key_value.detach().requires_grad_())
            # Manually add the gradients from later micro-sequences.
            key_value = AttachGrad.apply(key_value, present)

        if self._sequence_data_parallel_dim.group:
            key_value = (
                key_value[: kwargs[AttentionKwargs.sequence_k_dim].size]
                if sequence_first
                else key_value[:, : kwargs[AttentionKwargs.sequence_k_dim].size]
            )

        if sequence_first:
            # TODO: Optimize (is contiguous avoidable?)
            query = query.transpose(0, 1).contiguous()
            key_value = key_value.transpose(0, 1).contiguous()

        key, value = key_value.split(self._local_head_groups * self._config.head_size, dim=-1)

        query = query.view(*query.shape[:2], self._local_heads, self._config.head_size)
        key = key.view(*key.shape[:2], self._local_head_groups, self._config.head_size)
        value = value.view(*value.shape[:2], self._local_head_groups, self._config.head_size)

        if self._debug.enabled:
            self._debug(query, "query_rotary_input", self._query_dims, kwargs)
            self._debug(key, "key_rotary_input", self._kv_dims, kwargs)
        query, key = self._rotary(query, key, kwargs)

        window_size = (-1, -1) if self._config.window_size is None else (self._config.window_size - 1, 0)

        if self._use_flash_attention:
            assert _flash_available
            with set_generator(self._distributed.tp_generator):
                if (cu_seqlens_q := kwargs.get(AttentionKwargs.cu_seqlens_q, None)) is not None:
                    out_dims = query.size()
                    query = query.view(-1, query.size(-2), query.size(-1))
                    key = key.view(-1, key.size(-2), key.size(-1))
                    value = value.view(-1, value.size(-2), value.size(-1))
                    input_ = _flash_attn_varlen_func(
                        query,
                        key,
                        value,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=kwargs.get(AttentionKwargs.cu_seqlens_k),
                        max_seqlen_q=kwargs.get(AttentionKwargs.max_seqlen_q),
                        max_seqlen_k=kwargs.get(AttentionKwargs.max_seqlen_k),
                        dropout_p=self._config.dropout if self.training else 0.0,
                        window_size=window_size,
                        causal=True,
                        softmax_scale=self._softmax_scale,
                    ).view(*out_dims)
                else:
                    input_ = _flash_attn_func(
                        query,
                        key,
                        value,
                        window_size=window_size,
                        dropout_p=self._config.dropout if self.training else 0.0,
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
                kwargs[AttentionKwargs.attention_mask],
                kwargs[AttentionKwargs.attention_mask_value],
            )

        if self._debug.enabled:
            self._debug(query, "query", self._query_dims, kwargs)
            self._debug(key, "key", self._kv_dims, kwargs)
            self._debug(value, "value", self._kv_dims, kwargs)
            self._debug(input_, "context", self._context_dims, kwargs)

        if sequence_first:
            # TODO: Optimize (is contiguous avoidable? Transpose dense output?)
            input_ = input_.transpose(0, 1).contiguous()
        return self.dense(input_)

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        batch_dim: TensorDim = kwargs[AttentionKwargs.hidden_dims][1 if kwargs[AttentionKwargs.sequence_first] else 0]

        # Using this one since `hidden_dims` may be sequence-tensor-parallel, and attention is not.
        sequence_q_dim: TensorDim = kwargs[AttentionKwargs.sequence_q_dim]
        sequence_k_dim: TensorDim = kwargs[AttentionKwargs.sequence_k_dim]

        if config.global_:
            batch_size, sequence_q = batch_dim.global_size, sequence_q_dim.global_size
            # In case of sequence-data-parallel, we need to undo the shift in k-sequence-length.
            sequence_k = sequence_k_dim.global_size - sequence_q_dim.size * (
                sequence_q_dim.parallel_dim.size - sequence_q_dim.parallel_dim.rank - 1
            )
        else:
            batch_size, sequence_q = batch_dim.size, sequence_q_dim.size
            sequence_k = sequence_k_dim.size

        # 2 for multiply and accumulate, 2 operations (Q * K, attn * V), double for backward + Q * K recomputation.
        attn_compute_base = (
            2
            * (2 * config.forward + (5 if config.hardware else 4) * config.backward)
            * self._config.heads
            * self._config.head_size
        )

        if self._config.window_size is not None:
            # Remove the part of the past that lies completely outside the window, if applicable.
            sequence_k -= max(sequence_k - sequence_q - self._config.window_size, 0)

        attention_compute = sequence_q * sequence_k * attn_compute_base

        if (not config.hardware) or self._use_flash_attention:
            # Remove non-causal part. (TODO: Support non-causal)
            attention_compute -= (sequence_q * (sequence_q - 1) * attn_compute_base) // 2

            if self._config.window_size is not None:
                # Remove the part of the past that lies completely outside the window, if applicable.
                fully_out_of_window = max(sequence_k - sequence_q - self._config.window_size, 0)
                attention_compute -= fully_out_of_window * sequence_q * attn_compute_base
                # Remove the part of the past that lies partially outside the window, if applicable.
                partly_out_of_window = max(sequence_k - fully_out_of_window - self._config.window_size, 0)
                attention_compute -= (partly_out_of_window * (partly_out_of_window + 1) * attn_compute_base) // 2

        dense_input = TensorMeta.from_dims((batch_dim, sequence_q_dim, self._context_dims[-1]))

        # TODO: Add marginal compute? (ex. softmax)
        return sum(
            (
                self.query.get_compute_usage(input_, config),
                self.key_value.get_compute_usage(input_, config),
                attention_compute,
                self.dense.get_compute_usage(dense_input, config),
            )
        )
