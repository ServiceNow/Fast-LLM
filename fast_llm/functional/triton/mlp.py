import math
import typing

import torch

from fast_llm.core.distributed import ProcessGroup
from fast_llm.core.ops import gather_op
from fast_llm.functional.autograd import wrap_forward_backward
from fast_llm.functional.config import ActivationType, MLPRecomputeLevel, TritonConfig
from fast_llm.functional.linear import (
    input_parallel_linear_forward,
    maybe_transpose,
    output_parallel_linear_backward,
    output_parallel_linear_forward,
    update_linear_gradients,
)
from fast_llm.functional.triton import tl, tl_constexpr, triton_jit
from fast_llm.functional.triton.sparse_copy import (
    SparseMap,
    copy_dense_to_sparse_backward,
    copy_dense_to_sparse_forward,
    copy_sparse_to_dense_backward,
    copy_sparse_to_dense_forward,
)
from fast_llm.functional.triton.sparse_linear import output_sparse_matmul
from fast_llm.tensor import param_get_and_unset_is_zero

# Triton requires global variables to be annotated with `constexpr`.
_TritonActivationType: tl_constexpr = ActivationType


@triton_jit()
def triton_mlp_activation_forward_kernel(
    input_ptr,
    output_ptr,
    gated: tl_constexpr,
    activation_type: tl_constexpr,
    n_cols: tl_constexpr,
    block_size: tl_constexpr,
):
    # TODO: Int64 ptr only if needed?
    row_idx = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * block_size + tl.arange(0, block_size)

    output_offsets = n_cols * row_idx + columns
    input_offsets = 2 * n_cols * row_idx + columns if gated else output_offsets

    input_ptr = input_ptr + input_offsets
    mask = columns < n_cols

    input_ = tl.load(input_ptr, mask=mask).to(tl.float32)

    if activation_type == _TritonActivationType.gelu:
        tanh_input = 0.79788456 * input_ * (1 + 0.044715 * input_ * input_)
        tanh = 1 - 2 / (1 + tl.exp(2 * tanh_input))
        out = input_ * 0.5 * (1.0 + tanh)
    elif activation_type == _TritonActivationType.silu:
        out = input_ / (1 + tl.exp(-input_))
    elif activation_type == _TritonActivationType.relu:
        out = tl.where(input_ > 0, input_, 0)
    elif activation_type == _TritonActivationType.squared_relu:
        relu_out = tl.where(input_ > 0, input_, 0)
        out = relu_out * relu_out
    elif activation_type == _TritonActivationType.identity:
        out = input_
    else:
        tl.static_assert(False, activation_type)

    if gated:
        other = tl.load(input_ptr + n_cols, mask=mask)
        out = out * other

    tl.store(output_ptr + output_offsets, out, mask=mask)


@triton_jit()
def triton_mlp_activation_backward_kernel(
    grad_output_ptr,
    grad_input_ptr,
    input_ptr,
    output_ptr,
    gated: tl_constexpr,
    activation_type: tl_constexpr,
    recompute: tl_constexpr,
    n_cols: tl_constexpr,
    block_size: tl_constexpr,
):
    # TODO: Int64 ptr only if needed?
    row_idx = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * block_size + tl.arange(0, block_size)

    output_offsets = n_cols * row_idx + columns
    input_offsets = 2 * n_cols * row_idx + columns if gated else output_offsets

    input_ptr = input_ptr + input_offsets
    grad_input_ptr = grad_input_ptr + input_offsets

    mask = columns < n_cols

    input_ = tl.load(input_ptr, mask=mask).to(tl.float32)
    output_grad = tl.load(grad_output_ptr + output_offsets, mask=mask).to(tl.float32)

    if activation_type == _TritonActivationType.gelu:
        tanh_input = 0.79788456 * input_ * (1 + 0.044715 * input_ * input_)
        tanh = 1 - 2 / (1 + tl.exp(2 * tanh_input))
        grad = 0.5 * input_ * ((1 - tanh * tanh) * (0.79788456 + 0.1070322243 * input_ * input_)) + 0.5 * (1 + tanh)
        if gated or recompute:
            out = input_ * 0.5 * (1.0 + tanh)
    elif activation_type == _TritonActivationType.silu:
        exp = tl.exp(-input_)
        sigma = 1 / (1 + exp)
        grad = sigma * sigma + (1 + input_) / (2 + exp + 1 / exp)
        if gated or recompute:
            out = input_ * sigma
    elif activation_type == _TritonActivationType.relu:
        grad = tl.where(input_ > 0, 1, 0)
        if gated or recompute:
            out = tl.where(input_ > 0, input_, 0)
    elif activation_type == _TritonActivationType.squared_relu:
        relu_out = tl.where(input_ > 0, input_, 0)
        grad = 2 * relu_out
        if gated or recompute:
            out = relu_out * relu_out
    elif activation_type == _TritonActivationType.identity:
        grad = 1
        if gated or recompute:
            out = input_
    else:
        tl.static_assert(False, activation_type)

    if gated:
        other = tl.load(input_ptr + n_cols, mask=mask)
        tl.store(grad_input_ptr, grad * other * output_grad, mask=mask)
        tl.store(grad_input_ptr + n_cols, out * output_grad, mask=mask)  # noqa
        out = out * other
    else:
        tl.store(grad_input_ptr, grad * output_grad, mask=mask)

    if recompute:
        tl.store(output_ptr + output_offsets, out, mask=mask)  # noqa


def triton_mlp_activation_forward(
    input_: torch.Tensor,
    gated: bool,
    activation_type: ActivationType,
):
    # TODO: Improve assumptions.
    assert input_.is_contiguous()

    n_cols = input_.size(-1) // (2 if gated else 1)
    output = input_.new_empty(input_.shape[:-1] + (n_cols,))

    triton_mlp_activation_forward_kernel[
        (output.numel() // n_cols, math.ceil(n_cols / TritonConfig.POINTWISE_BLOCK_SIZE))
    ](
        input_,
        output,
        gated=gated,  # noqa
        activation_type=activation_type.value,  # noqa
        n_cols=n_cols,  # noqa
        block_size=TritonConfig.POINTWISE_BLOCK_SIZE,
    )
    return output, (input_, gated, activation_type)


def triton_mlp_activation_backward(grad_output: torch.Tensor, context: tuple, recompute: bool = False):
    # TODO: Improve assumptions.
    assert grad_output.is_contiguous()

    input_, gated, activation_type = context
    grad_input = torch.empty_like(input_)
    output = torch.empty_like(grad_output) if recompute else None

    n_cols = grad_output.size(-1)

    triton_mlp_activation_backward_kernel[
        (grad_output.numel() // n_cols, math.ceil(n_cols / TritonConfig.POINTWISE_BLOCK_SIZE))
    ](
        grad_output,
        grad_input,
        input_,
        output,
        gated=gated,
        activation_type=activation_type,
        recompute=recompute,  # noqa
        n_cols=n_cols,  # noqa
        block_size=TritonConfig.POINTWISE_BLOCK_SIZE,
    )
    return grad_input, output


triton_mlp_activation_autograd = wrap_forward_backward(triton_mlp_activation_forward, triton_mlp_activation_backward)


def torch_mlp_activation(
    input_: torch.Tensor,
    gated: bool,
    activation_type: ActivationType,
) -> torch.Tensor:
    if gated:
        x1, x2 = input_.chunk(2, dim=-1)
        return activation_type.activation_fn(x1) * x2
    else:
        return activation_type.activation_fn(input_)


def mlp_forward(
    input_: torch.Tensor,
    scores: torch.Tensor | None,
    weight_1: torch.Tensor,
    bias_1: torch.Tensor | None,
    weight_2: torch.Tensor,
    bias_2: torch.Tensor | None,
    gated: bool,
    activation_type: ActivationType,
    group: ProcessGroup | None,
    sequence_parallel: bool,
    training: bool = True,
    recompute_level: MLPRecomputeLevel = MLPRecomputeLevel.none,
    transposed_layer_2_weight: bool = False,
    sparse_map: SparseMap | None = None,
) -> tuple[torch.Tensor, list[typing.Any] | None]:
    # Sparse copy
    input_shape = input_.shape
    intermediate_0 = input_ if sparse_map is None else copy_dense_to_sparse_forward(input_, sparse_map)[0]

    # Layer 1
    intermediate_1, _ = output_parallel_linear_forward(
        intermediate_0, weight_1, bias_1, group, sequence_parallel, False, sparse_map
    )

    if recompute_level.recompute_sparse_input:
        intermediate_0 = None
    else:
        input_ = None

    # Activation
    if TritonConfig.TRITON_ENABLED:
        intermediate_2, _ = triton_mlp_activation_forward(intermediate_1, gated, activation_type)
    else:
        do_grad = training and not recompute_level.recompute_activation
        with torch.set_grad_enabled(do_grad):
            intermediate_2 = torch_mlp_activation(
                intermediate_1.detach().requires_grad_(do_grad), gated, activation_type
            )
    if recompute_level.recompute_layer_1:
        intermediate_1 = None

    # Layer 2
    intermediate_3, _ = input_parallel_linear_forward(
        intermediate_2,
        weight_2,
        bias_2,
        group,
        sequence_parallel,
        transposed_layer_2_weight,
        sparse_map,
    )

    # Context
    if recompute_level.recompute_activation or not training:
        intermediate_2 = None

    # Sparse copy
    if sparse_map is None:
        output = intermediate_3
        intermediate_3 = None
    else:
        output, _ = copy_sparse_to_dense_forward(intermediate_3, scores, sparse_map)

    context = (
        [
            input_,
            scores,
            intermediate_0,
            intermediate_1,
            intermediate_2,
            intermediate_3,
            weight_1,
            bias_1,
            weight_2,
            bias_2,
            gated,
            activation_type,
            group,
            sequence_parallel,
            transposed_layer_2_weight,
            sparse_map,
            input_shape,
        ]
        if training
        else None
    )
    return output, context


def mlp_backward(grad_output: torch.Tensor, context: list[typing.Any]) -> tuple[torch.Tensor, torch.Tensor]:
    (
        input_,
        scores,
        intermediate_0,
        intermediate_1,
        intermediate_2,
        intermediate_3,
        weight_1,
        bias_1,
        weight_2,
        bias_2,
        gated,
        activation_type,
        group,
        sequence_parallel,
        transposed_layer_2_weight,
        sparse_map,
        input_shape,
    ) = context
    context.clear()

    # Sparse copy
    if sparse_map is None:
        grad_scores = None
    else:
        grad_output, grad_scores = copy_sparse_to_dense_backward(grad_output, (sparse_map, intermediate_3, scores))

    # Gather sequence-parallel slices (non-overlapped; from input_parallel_backward)
    if sequence_parallel:
        grad_output = gather_op(grad_output, group, dim=0)

    # Layer 2 input grad
    weight_2_t = maybe_transpose(weight_2, transposed_layer_2_weight)
    if sparse_map is None:
        grad_intermediate_2 = grad_output.matmul(weight_2_t)
    else:
        grad_intermediate_2 = output_sparse_matmul(grad_output, weight_2_t, sparse_map)

    # Sparse input recomputation
    if intermediate_0 is None:
        intermediate_0 = input_ if sparse_map is None else copy_dense_to_sparse_forward(input_, sparse_map)[0]

    # Layer 1 recomputation
    if intermediate_1 is None:
        intermediate_1 = output_parallel_linear_forward(
            intermediate_0, weight_1, bias_1, group, sequence_parallel, False, sparse_map
        )[0]

    # Activation recomputation and/or backward
    if TritonConfig.TRITON_ENABLED:
        grad_intermediate_1, intermediate_2_ = triton_mlp_activation_backward(
            grad_intermediate_2, (intermediate_1, gated, activation_type), intermediate_2 is None
        )
    else:
        if intermediate_2 is None:
            with torch.set_grad_enabled(True):
                intermediate_2_ = torch_mlp_activation(
                    intermediate_1.detach().requires_grad_(True), gated, activation_type
                )
        else:
            intermediate_2_ = intermediate_2
        intermediate_2_.backward(grad_intermediate_2)
        grad_intermediate_1 = intermediate_1.grad

    # Layer 2 parameter grad
    del grad_intermediate_2, intermediate_1
    update_linear_gradients(
        intermediate_2_ if intermediate_2 is None else intermediate_2,
        weight_2,
        bias_2,
        grad_output,
        transposed_layer_2_weight,
        sparse_map,
    )
    del grad_output, intermediate_2, intermediate_2_

    # Layer 1 backward
    grad_input = output_parallel_linear_backward(
        grad_intermediate_1,
        (intermediate_0, weight_1, bias_1, group, sequence_parallel, False, sparse_map),
    )

    # Sparse copy
    if sparse_map is not None:
        grad_input = copy_dense_to_sparse_backward(grad_input, (sparse_map, input_shape))

    return grad_input, grad_scores


mlp_autograd = wrap_forward_backward(mlp_forward, mlp_backward)


class ChunkWeight(torch.autograd.Function):
    """
    Chunk a weight without letting autograd know about it, i.e., make it believe it's actually separate weights.
    Must be associated with `ChunkWeightPost`.
    TODO: Would be simpler to define custom forward and backward for looped mlp instead?
    """

    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, weight: torch.Tensor, num_chunks: int
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:  # noqa
        with torch.no_grad():
            weight_chunked = weight.chunk(num_chunks)
            grad_buffer_chunked = weight.grad_buffer.chunk(num_chunks)  # noqa
            for weight_chunk, grad_buffer_chunk in zip(weight_chunked, grad_buffer_chunked):
                weight_chunk.requires_grad_(weight.requires_grad)
                weight_chunk.grad_buffer = grad_buffer_chunk
        ctx.weight = weight
        ctx.weight_chunked = weight_chunked
        return input_, weight_chunked  # OK?

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, dummy) -> tuple[torch.Tensor, None, None]:  # noqa
        for weight_chunk in ctx.weight_chunked:
            # Check for runaway grads.
            assert weight_chunk.grad is None
            # Make sure all grads are computed.
            if weight_chunk.param_grad_is_zero:
                weight_chunk.grad_buffer.zero_()
        return grad_input, None, None


chunk_weight = ChunkWeight.apply


class ChunkWeightPost(torch.autograd.Function):
    """
    Autograd hook with `ChunkWeight` called after all uses of the weight, so gradient accumulation is handled correctly.
    TODO: Would be simpler to define custom forward and backward for looped mlp instead?
    """

    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, weight: torch.Tensor, weight_chunked: tuple[torch.Tensor]
    ) -> torch.Tensor:  # noqa
        ctx.weight, ctx.weight_chunked = weight, weight_chunked
        return input_

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor) -> tuple[torch.Tensor, None, None]:  # noqa
        is_zero = param_get_and_unset_is_zero(ctx.weight)
        for weight_chunk in ctx.weight_chunked:
            weight_chunk.param_grad_is_zero = is_zero
        return grad_input, None, None


chunk_weight_post = ChunkWeightPost.apply


def mlp_autograd_looped(
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    top_experts: torch.Tensor,
    weight_1: torch.Tensor,
    weight_2: torch.Tensor,
    num_experts: int,
    gated: bool,
    activation_type: ActivationType,
    group: ProcessGroup | None,
    sequence_parallel: bool,
    training: bool = True,
    recompute_level: MLPRecomputeLevel = MLPRecomputeLevel.none,
) -> torch.Tensor:
    # TODO: Needed?
    scores = scores.to(hidden_states.dtype)
    expert_mask = torch.nn.functional.one_hot(top_experts, num_classes=num_experts).permute(2, 1, 0)
    output = torch.zeros_like(hidden_states)

    hidden_states, weight_1_chunked = chunk_weight(hidden_states, weight_1, num_experts)
    hidden_states, weight_2_t_chunked = chunk_weight(hidden_states, weight_2, num_experts)

    for expert_idx, (weight_1_chunk, weight_2_t_chunk) in enumerate(zip(weight_1_chunked, weight_2_t_chunked)):
        row, column = torch.where(expert_mask[expert_idx])
        if column.size(0) > 0:
            output[column] += (
                mlp_autograd(
                    hidden_states[column],
                    None,
                    weight_1_chunk,
                    None,
                    weight_2_t_chunk,
                    None,
                    gated,
                    activation_type,
                    group,
                    sequence_parallel,
                    training,
                    recompute_level,
                    True,
                )
                * scores[column, row, None]
            )

    output = chunk_weight_post(output, weight_2, weight_2_t_chunked)
    output = chunk_weight_post(output, weight_1, weight_1_chunked)

    return output
