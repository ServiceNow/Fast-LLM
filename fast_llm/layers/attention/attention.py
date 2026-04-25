import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import gather_op, reduce_op, reduce_scatter_op
from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.engine.config_utils.tensor_dim import CompositeTensorDim, ConcatenatedTensorDim, TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.functional.utils import wrap_forward_backward
from fast_llm.layers.attention.config import AttentionConfig, AttentionImplementation, AttentionKwargs
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.tensor import TensorMeta
from fast_llm.utils import div

try:
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # noqa
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as _flash_attn_varlen_func

    _flash_available = torch.cuda.is_available()
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

    # Preprocessing
    _backup_attention_mask: torch.Tensor
    _backup_attention_mask_value: torch.Tensor
    _backup_attention_tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_bias: bool = True,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
            return_bias=return_bias,
        )
        self._implementation = self._config.implementation
        if self._implementation == AttentionImplementation.auto:
            if _flash_available and self._distributed_config.compute_dtype in (DataType.float16, DataType.bfloat16):
                self._implementation = AttentionImplementation.flash
            else:
                self._implementation = AttentionImplementation.backup

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
        key_dim = CompositeTensorDim("key", (head_group_dim, head_size_dim))
        key_value_dim = (
            key_dim
            if self._config.attention_k_eq_v
            else ConcatenatedTensorDim(
                "key_value",
                (
                    key_dim,
                    CompositeTensorDim("value", (head_group_dim, head_size_dim)),
                ),
            )
        )
        self._dense_dim = CompositeTensorDim("dense", (head_group_dim, group_heads_dim, head_size_dim))

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
        if self._peft is not None and self._config.key_layer.apply_peft is None and not self._config.attention_k_eq_v:
            # Default: Apply to value only.
            # TODO: Avoid this hack.
            self.key_value = self._peft.apply_linear(
                self.key_value, True, out_channel_begin=div(key_value_dim.global_size, 2)
            )

        self._query_key_value = wrap_forward_backward(self._query_key_value_forward, self._query_key_value_backward)

        # Rotary embeddings.
        self._rotary = self._config.rotary.get_layer(head_size_dim)

        # Per-head QK norms (applied after projection, before RoPE).
        self.q_norm = (
            self._config.query_norm.get_layer(head_size_dim, lr_scale=self._lr_scale, peft=self._peft)
            if self._config.query_norm is not None
            else None
        )
        self.k_norm = (
            self._config.key_norm.get_layer(head_size_dim, lr_scale=self._lr_scale, peft=self._peft)
            if self._config.key_norm is not None
            else None
        )

        # Output.
        self.dense = self._config.dense_layer.get_layer(
            self._dense_dim,
            hidden_dim,
            default_weight_initialization=init_normal_(std=self._hidden_size**-0.5),
            default_add_bias=self._config.add_linear_biases,
            sequence_parallel=self._sequence_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        # Debug dims
        self._query_dims = (
            CompositeTensorDim("heads", (head_group_dim, group_heads_dim)),
            head_size_dim,
        )
        self._kv_dims = (
            head_group_dim,
            head_size_dim,
        )

    def _attn_backup(
        self,
        query: torch.Tensor,  # sq, head_per_group * head_group, head_size
        key: torch.Tensor,  # sk, head_group, head_size
        value: torch.Tensor,  # sk, head_group, head_size
        kwargs: dict[str, typing.Any],
    ) -> torch.Tensor:  # sq, head_per_group * head_group, head_size
        # Backup attention (inefficient)
        sq = query.size(0)
        sk = key.size(0)

        # sq, head_per_group * head_group, head_size -> head_group, sq * head_per_group, head_size
        query = (
            query.unflatten(1, (self._local_head_groups, self._local_heads_per_group))
            .transpose(0, 1)
            .reshape(self._local_head_groups, sq * self._local_heads_per_group, self._config.head_size)
        )
        # sk, head_group, head_size -> head_group, head_size, sk
        key = key.movedim(0, 2)
        # sk, head_group, head_size -> head_group, sk, head_size
        value = value.transpose(0, 1)

        attn_weights = torch.empty(
            (self._local_head_groups, sq * self._local_heads_per_group, sk), device=query.device, dtype=query.dtype
        )
        attn_weights = torch.baddbmm(
            attn_weights,
            query,
            key,
            beta=0,
            alpha=self._softmax_scale,
        ).view(self._local_head_groups, sq, self._local_heads_per_group, sk)

        attn_weights = attn_weights.to(torch.float32)
        if (attention_mask := kwargs[AttentionKwargs.attention_mask]) is not None:
            attn_weights = torch.where(attention_mask, attn_weights, kwargs[AttentionKwargs.attention_mask_value])
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(query.dtype)

        attn_weights = torch.dropout(attn_weights, self._config.dropout, self.training)
        attn_output = torch.bmm(
            attn_weights.view(self._local_head_groups, sq * self._local_heads_per_group, sk).to(value.dtype), value
        )
        # head_group, sq * head_per_group, head_size -> sq, head_per_group * head_group, head_size
        return (
            attn_output.view(self._local_head_groups, sq, self._local_heads_per_group, self._config.head_size)
            .transpose(0, 1)
            .flatten(1, 2)
        )

    def _attn_flash(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kwargs: dict[str, typing.Any]
    ) -> torch.Tensor:
        assert _flash_available
        return _flash_attn_varlen_func(
            query,
            key,
            value,
            kwargs[AttentionKwargs.cu_seqlens_q],
            kwargs[AttentionKwargs.cu_seqlens_k],
            kwargs[AttentionKwargs.max_seqlen_q],
            kwargs[AttentionKwargs.max_seqlen_k],
            dropout_p=self._config.dropout if self.training else 0.0,
            window_size=(-1, -1) if self._config.window_size is None else (self._config.window_size - 1, 0),
            causal=self._config.causal,
            softmax_scale=self._softmax_scale,
        )

    def _query_key_value_forward(
        self, input_: torch.Tensor, kwargs: dict[str, typing.Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, typing.Any]]:
        key_value, key_value_context = self.key_value.forward_only(input_)

        handle = None

        if self._config.head_groups == 1 and self._sequence_parallel:
            key_value, handle = gather_op(key_value, group=self._parallel_dim.group, dim=0, async_op=True)

        query, query_context = self.query.forward_only(input_)

        if handle:
            # TODO: This is probably unnecessary.
            handle.wait()

        query = query.unflatten(1, (self._local_heads, self._config.head_size))
        if self._config.attention_k_eq_v:
            key_value = key_value.unflatten(1, (self._local_head_groups, self._config.head_size))
            key_value = torch.cat([key_value, key_value], dim=1)
        else:
            key_value = key_value.unflatten(1, (2 * self._local_head_groups, self._config.head_size))

        # QK norms: applied after projection/unflatten, before RoPE.
        # wrap_forward_backward disables autograd inside forward_only, so we re-enable it
        # locally to build a mini-graph for each norm; the saved (pre, out) pair is used
        # in _query_key_value_backward to call .backward() and accumulate weight grads.
        norm_context = {}
        if self.q_norm is not None:
            # .contiguous() needed: unflatten returns a non-contiguous view and
            # Normalization.forward uses .view() internally.
            # .detach() before passing onward keeps the mini-graph alive only in
            # context; without it, Function.forward returning query_normed causes
            # PyTorch to replace its grad_fn with the outer Function node, which
            # would re-enter _query_key_value_backward when we call .backward() here.
            query_pre = query.contiguous().detach().requires_grad_(True)
            with torch.enable_grad():
                query_normed = self.q_norm(query_pre)
            norm_context["q_norm"] = (query_pre, query_normed)
            # Triton rotary is in-place. Clone to keep the q-norm mini-graph output
            # intact for the manual backward call below.
            query = query_normed.detach().clone()
        if self.k_norm is not None or self._config.value_norm:
            k, v = key_value.chunk(2, dim=1)
            if self.k_norm is not None:
                k_pre = k.contiguous().detach().requires_grad_(True)
                with torch.enable_grad():
                    k_normed = self.k_norm(k_pre)
                norm_context["k_norm"] = (k_pre, k_normed)
                k = k_normed.detach()
            if self._config.value_norm:
                v_pre = v.contiguous().detach().requires_grad_(True)
                with torch.enable_grad():
                    v_normed = torch.rms_norm(
                        v_pre, (self._config.head_size,), None, self._config.value_norm_eps
                    )
                norm_context["v_norm"] = (v_pre, v_normed)
                v = v_normed.detach()
            key_value = torch.cat([k, v], dim=1)

        query, key_value, rotary_context = self._rotary.forward_only(query, key_value, kwargs)

        if self._sequence_data_parallel_dim.group:
            # sequence dim may not be zero, but this needs to be handled after `handle.wait()`
            key_value, handle = gather_op(
                key_value, group=self._sequence_data_parallel_dim.group, dim=0, async_op=True
            )
        if handle:
            handle.wait()

        context = {"query": query_context, "key_value": key_value_context, "rotary": rotary_context, **norm_context}
        return query, key_value, context

    def _query_key_value_backward(
        self, query_grad: torch.Tensor, key_value_grad: torch.Tensor, context: dict
    ) -> torch.Tensor:
        # TODO: De-allocate qkv grads quicker.
        key_value_grad, handle = reduce_scatter_op(
            key_value_grad,
            group=self._sequence_data_parallel_dim.group,
            dim=0,
            async_op=True,
        )

        rotary_context = context.pop("rotary")
        query_grad, _ = self._rotary.backward(query_grad, None, rotary_context)

        if q_norm_ctx := context.pop("q_norm", None):
            query_pre, query_normed = q_norm_ctx
            query_normed.backward(query_grad)
            query_grad = query_pre.grad

        # TODO: Overlap with both.
        input_grad = self.query.backward(query_grad.flatten(1), context.pop("query"))

        if handle:
            handle.wait()

        _, key_value_grad = self._rotary.backward(None, key_value_grad, rotary_context)

        if "k_norm" in context or "v_norm" in context:
            k_grad, v_grad = key_value_grad.chunk(2, dim=1)
            if k_norm_ctx := context.pop("k_norm", None):
                k_pre, k_normed = k_norm_ctx
                k_normed.backward(k_grad.contiguous())
                k_grad = k_pre.grad
            if v_norm_ctx := context.pop("v_norm", None):
                v_pre, v_normed = v_norm_ctx
                v_normed.backward(v_grad.contiguous())
                v_grad = v_pre.grad
            key_value_grad = torch.cat([k_grad, v_grad], dim=1)

        if self._config.attention_k_eq_v:
            k_grad, v_grad = key_value_grad.chunk(2, dim=1)
            key_value_grad = (k_grad + v_grad).flatten(1)
        else:
            key_value_grad = key_value_grad.flatten(1)

        if self._config.head_groups == 1 and (group := self._parallel_dim.group):
            if self._sequence_parallel:
                key_value_grad = reduce_scatter_op(key_value_grad, group=group, dim=0)
            else:
                key_value_grad = reduce_op(key_value_grad, group=group)

        input_grad.add_(self.key_value.backward(key_value_grad, context.pop("key_value")))
        return input_grad

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._debug(input_, "attn_input", (kwargs[AttentionKwargs.hidden_token_dim], self._hidden_dim), kwargs)
        query, key_value = self._query_key_value(input_, kwargs)

        self._debug(
            key_value.chunk(2, dim=1)[0],
            "key_rotary_input",
            (kwargs[AttentionKwargs.key_value_token_dim], *self._kv_dims),
            kwargs,
        )

        # TODO: These get unnecessarily big with lots of small documents.
        if (past_key_values := kwargs.get(AttentionKwargs.past_key_values)) is not None:
            # Clear the lists so tensors can be de-allocated
            key_value = torch.cat((past_key_values.pop(0), key_value), dim=0)

        if (presents := kwargs.get(AttentionKwargs.presents)) is not None:
            # Return the presents as a leaf tensors so the gradients from later micro-sequences
            # don't propagate to this one.
            presents.append(present := key_value.detach().requires_grad_())
            # Manually add the gradients from later micro-sequences.
            key_value = AttachGrad.apply(key_value, present)

        key_value = key_value[: kwargs[AttentionKwargs.sequence_k_dim].size]
        key, value = key_value.chunk(2, dim=1)

        with set_generator(self._distributed.tp_generator):
            if self._implementation == AttentionImplementation.flash:
                input_ = self._attn_flash(query, key, value, kwargs)
            elif self._implementation == AttentionImplementation.backup:
                # TODO: Avoid the flattens.
                input_ = self._attn_backup(query, key, value, kwargs)
            else:
                raise NotImplementedError(self._implementation)
        input_ = input_.flatten(1)
        self._debug(query, "query", (token_dim := kwargs[AttentionKwargs.token_dim], *self._query_dims), kwargs)
        self._debug(key, "key", (sequence_k_dim := kwargs[AttentionKwargs.sequence_k_dim], *self._kv_dims), kwargs)
        self._debug(value, "value", (sequence_k_dim, *self._kv_dims), kwargs)
        self._debug(input_, "context", (token_dim, self._dense_dim), kwargs)

        out, bias = self.dense(input_)
        self._debug(
            out,
            None,
            (
                token_dim,
                self._hidden_dim,
            ),
            kwargs,
        )
        return out, bias

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Account for varlen
        sequence_q_dim: TensorDim = kwargs[AttentionKwargs.token_dim]
        sequence_k_dim: TensorDim = kwargs[AttentionKwargs.sequence_k_dim]

        if config.global_:
            sequence_q = sequence_q_dim.global_size
            # In case of sequence-data-parallel, we need to undo the shift in k-sequence-length.
            sequence_k = sequence_k_dim.global_size - sequence_q_dim.size * (
                sequence_q_dim.parallel_dim.size - sequence_q_dim.parallel_dim.rank - 1
            )
        else:
            sequence_q = sequence_q_dim.size
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

        if (not config.hardware) or self._implementation in AttentionImplementation.flash:
            # Remove non-causal part. (TODO: Support non-causal)
            # TODO: Compute is overestimated without cross-document attention.
            attention_compute -= (sequence_q * (sequence_q - 1) * attn_compute_base) // 2

            if self._config.window_size is not None:
                # Remove the part of the past that lies completely outside the window, if applicable.
                fully_out_of_window = max(sequence_k - sequence_q - self._config.window_size, 0)
                attention_compute -= fully_out_of_window * sequence_q * attn_compute_base
                # Remove the part of the past that lies partially outside the window, if applicable.
                partly_out_of_window = max(sequence_k - fully_out_of_window - self._config.window_size, 0)
                attention_compute -= (partly_out_of_window * (partly_out_of_window + 1) * attn_compute_base) // 2

        dense_input = TensorMeta.from_dims((*input_.dims[:-1], self._dense_dim))

        # TODO: Add marginal compute? (ex. softmax)
        return sum(
            (
                self.query.get_compute_usage(input_, config),
                self.key_value.get_compute_usage(input_, config),
                attention_compute,
                self.dense.get_compute_usage(dense_input, config),
            )
        )

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return (
            {
                "return_cumulative_sequence_lengths": True,
                "return_max_sequence_lengths": True,
                "causal": self._config.causal,
            }
            if self._implementation == AttentionImplementation.flash
            else {"return_document_index": True, "causal": self._config.causal}
        )

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        self._rotary.preprocess(kwargs)
        if self._implementation == AttentionImplementation.backup:
            self._preprocess_for_backup_attention(kwargs)

    def _preprocess_for_backup_attention(self, kwargs: dict[str, typing.Any]) -> None:
        device = kwargs[AttentionKwargs.device] if AttentionKwargs.device in kwargs else self._distributed.device
        sequence_k = kwargs[AttentionKwargs.sequence_k_dim].size
        sequence_q = kwargs[AttentionKwargs.token_dim].size
        if self._config.causal:
            if (
                sequence_length := kwargs[AttentionKwargs.sequence_length]
            ) > self._backup_attention_tensor_cache_max_sequence_length:
                # Create tensor cache.
                self._backup_attention_tensor_cache_max_sequence_length = sequence_length

                self._backup_attention_mask = torch.ones(
                    (sequence_length, sequence_length),
                    dtype=torch.bool,
                    device=device,
                ).tril_()

                if self._config.window_size is not None:
                    self._backup_attention_mask.triu_(-self._config.window_size + 1)
            attention_mask = self._backup_attention_mask[None, sequence_k - sequence_q : sequence_k, None, :sequence_k]
        else:
            attention_mask = None

        document_mask = (
            kwargs[AttentionKwargs.document_index_k][None, None, None, :]
            == kwargs[AttentionKwargs.document_index_q][None, :, None, None]
        )
        if attention_mask is None:
            attention_mask = document_mask
        else:
            attention_mask = attention_mask & document_mask

        kwargs[AttentionKwargs.attention_mask] = attention_mask

        if attention_mask is not None:
            if not hasattr(self, "_backup_attention_mask_value"):
                self._backup_attention_mask_value = torch.full(
                    [],
                    torch.finfo(self._distributed_config.compute_dtype.torch).min,
                    dtype=self._distributed_config.compute_dtype.torch,
                    device=device,
                )
            kwargs[AttentionKwargs.attention_mask_value] = self._backup_attention_mask_value
