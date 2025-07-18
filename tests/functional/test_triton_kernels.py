import pytest
import torch

from fast_llm.functional.config import (
    MAX_DROPLESS_BLOCK_SIZE_ROW,
    ActivationType,
    CrossEntropyImpl,
    TargetFormat,
    TritonConfig,
)
from fast_llm.functional.cross_entropy import cross_entropy_forward_backward
from fast_llm.functional.triton.adam import triton_adam
from fast_llm.functional.triton.mlp import (
    torch_mlp_activation,
    triton_mlp_activation_backward,
    triton_mlp_activation_forward,
)
from fast_llm.functional.triton.normalization import (
    rms_norm,
    triton_normalization_backward,
    triton_normalization_forward,
)
from fast_llm.functional.triton.pointwise import triton_add, triton_copy, triton_fill
from fast_llm.functional.triton.rotary import triton_rotary_
from fast_llm.functional.triton.sparse_copy import get_sparse_map
from fast_llm.layers.transformer.rotary.config import DefaultRotaryConfig
from fast_llm.layers.transformer.rotary.rotary import (
    apply_rotary_embeddings,
    convert_rotary_complex_to_real,
    convert_rotary_real_to_complex,
)
from fast_llm.utils import Assert, rms_diff
from tests.utils.utils import requires_cuda


@requires_cuda
def test_triton_fill():
    assert TritonConfig.TRITON_ENABLED
    x = torch.randn(425, 549, dtype=torch.bfloat16, device="cuda")
    triton_fill(x, 32)
    assert x.min().item() == x.max().item() == 32


@requires_cuda
def test_triton_copy():
    assert TritonConfig.TRITON_ENABLED
    x = torch.randn(7563, dtype=torch.bfloat16, device="cuda")
    x1 = x.clone()
    y = torch.zeros_like(x)
    Assert.all_different(x, y)
    triton_copy(x, y)
    Assert.all_equal(x, y)
    Assert.all_equal(x, x1)


@requires_cuda
def test_triton_copy_cast():
    assert TritonConfig.TRITON_ENABLED
    x = torch.randn(7563, dtype=torch.bfloat16, device="cuda")
    x1 = x.clone()
    y = torch.zeros_like(x, dtype=torch.float32)
    Assert.all_different(x.float(), y)
    triton_copy(x, y)
    Assert.rms_close(x, y, 1e-4)
    Assert.all_equal(x, x1)


@requires_cuda
def test_triton_add():
    assert TritonConfig.TRITON_ENABLED
    x = torch.randn(8934, dtype=torch.float32, device="cuda")
    x1 = x.clone()
    y = torch.zeros_like(x)
    y1 = y.clone()
    Assert.all_different(x, y)
    z = triton_add(x, y)
    z1 = x1 + y1
    Assert.rms_close(z, z1, 1e-5)
    Assert.all_equal(x, x1)
    Assert.all_equal(y, y1)


@requires_cuda
@pytest.mark.parametrize(
    ("batch_size", "sequence_length", "num_heads", "kv_channels"),
    [(4, 1024, 8, 128), (1, 32, 1, 16), (2, 2048, 2, 192), (3, 519, 7, 134)],
)
def test_triton_rotary(batch_size, sequence_length, num_heads, kv_channels):
    assert TritonConfig.TRITON_ENABLED
    x = torch.randn(batch_size, sequence_length, num_heads, kv_channels, dtype=torch.bfloat16, device="cuda")

    y1 = apply_rotary_embeddings(
        x,
        DefaultRotaryConfig(triton=False)
        .build()
        ._get_frequencies(
            sequence_length,
            kv_channels,
            device="cuda",
        ),
    )

    y2 = convert_rotary_real_to_complex(
        triton_rotary_(
            convert_rotary_complex_to_real(x, kv_channels, 3),
            DefaultRotaryConfig(triton=True).build()._get_frequencies(sequence_length, kv_channels, device="cuda"),
        ),
        kv_channels,
        3,
    )
    Assert.rms_close(y1, y2, 1e-3)


@requires_cuda
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("zero_centered", [True, False])
def test_triton_normalization(has_bias, zero_centered):
    assert TritonConfig.TRITON_ENABLED
    input_ = torch.randn(4096, 1024, device="cuda", requires_grad=True)
    output_grad = torch.randn_like(input_)

    weight = torch.randn(1024, device="cuda", requires_grad=True)
    weight.grad_buffer = torch.empty_like(weight)
    weight.param_grad_is_zero = True

    if has_bias:
        bias = torch.randn_like(weight, requires_grad=True)
        bias.grad_buffer = torch.empty_like(bias)
        bias.param_grad_is_zero = True
    else:
        bias = None

    output0, context = triton_normalization_forward(input_, weight, bias, 1e-5, True, zero_centered)
    input_grad0 = triton_normalization_backward(output_grad, context)
    weight_grad0 = weight.grad_buffer.clone()
    weight.param_grad_is_zero = True
    if has_bias:
        bias_grad0 = bias.grad_buffer.clone()
        bias.param_grad_is_zero = True

    # Check reproducibility
    output1, context = triton_normalization_forward(input_, weight, bias, 1e-5, True, zero_centered)
    input_grad1 = triton_normalization_backward(output_grad, context)
    Assert.all_equal(output0, output1)
    Assert.all_equal(input_grad0, input_grad1)
    Assert.all_equal(weight_grad0, weight.grad_buffer)
    if has_bias:
        Assert.all_equal(bias_grad0, bias.grad_buffer)

    # Compare to other implementation
    if has_bias:
        output2 = torch.nn.functional.layer_norm(input_, weight.shape, weight + zero_centered, bias, 1e-5)
    else:
        output2 = rms_norm(input_, weight + zero_centered, 1e-5)
    output2.backward(output_grad)

    Assert.rms_close(output0, output2, 1e-5)
    Assert.rms_close(input_grad0, input_.grad, 1e-5)
    Assert.rms_close(weight_grad0, weight.grad, 1e-3)
    if has_bias:
        Assert.rms_close(bias_grad0, bias.grad, 1e-3)


@requires_cuda
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize(
    "activation_type",
    [
        ActivationType.gelu,
        ActivationType.silu,
        ActivationType.relu,
        ActivationType.squared_relu,
        ActivationType.identity,
    ],
)
@pytest.mark.parametrize("recompute", [True, False])
def test_triton_mlp_activation(gated, activation_type, recompute):
    assert TritonConfig.TRITON_ENABLED
    input_ = torch.randn(1024, 4096 * (2 if gated else 1), device="cuda", requires_grad=True)
    output_grad = torch.randn(1024, 4096, device="cuda")

    output1, context = triton_mlp_activation_forward(input_, gated, activation_type)
    input_grad1, output3 = triton_mlp_activation_backward(output_grad, context, recompute)

    output2 = torch_mlp_activation(input_, gated, activation_type)
    output2.backward(output_grad)

    Assert.rms_close(output1, output2, 1e-5)
    Assert.rms_close(input_grad1, input_.grad, 1e-5)
    if recompute:
        Assert.rms_close(output1, output3, 1e-5)


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize(
    ("num_columns", "grad_output", "logits_scale_factor", "loss_masking"),
    (
        (8192, 1.0, 1.0, False),  # Simple
        (5000, 1.0, 1.0, False),  # Not a power of 2
        (5000, None, 1.0, False),  # No grad
        (5000, 1.0, 4.0, False),  # Loss scaling
        (5000, 4.0, 1.0, False),  # Grad scaling
        (5000, 1.0, 1.0, True),  # Loss masking
        (65536, 1.0, 1.0, False),  # Max block size
        (65537, 1.0, 1.0, False),  # Above max block size
    ),
)
@pytest.mark.parametrize("target_format", (TargetFormat.labels, TargetFormat.logits, TargetFormat.probabilities))
def test_cross_entropy(num_columns, grad_output, logits_scale_factor, loss_masking, target_format):
    # TODO: Test tensor-parallel implementation.
    assert TritonConfig.TRITON_ENABLED
    # We want something moderately close to the target for the test to be meaningful
    logits_var = torch.randn(256, num_columns, dtype=torch.bfloat16, device="cuda") / 3
    loss_mask = torch.randint(0, 2, (256,), dtype=torch.bool, device="cuda") if loss_masking else None
    if target_format == TargetFormat.labels:
        target = torch.randint(0, num_columns, (256,), dtype=torch.int64, device="cuda")
        logits = (torch.nn.functional.one_hot(target, num_columns) + logits_var).requires_grad_()
        if loss_masking:
            logits = torch.where(loss_mask.unsqueeze(-1), logits, -100)
            loss_mask = None
    else:
        target = torch.randn(256, num_columns, dtype=torch.bfloat16, device="cuda")
        logits = (target + logits_var).requires_grad_()
        if target_format == TargetFormat.probabilities:
            target = torch.softmax(target, -1)

    kwargs = {
        "logits": logits,
        "target": target,
        "loss_mask": loss_mask,
        "grad_output": grad_output,
        "logits_scale_factor": logits_scale_factor,
        "target_format": target_format,
    }
    # Torch serves as the reference implementation.
    out_torch, grad_torch = cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.torch)

    out_fused, grad_fused = cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.fused)
    Assert.rms_close(out_fused, out_torch, 5e-3)
    if grad_output is None:
        assert grad_torch is None
        assert grad_fused is None
    else:
        Assert.rms_close(grad_fused, grad_torch, 5e-3)

    if num_columns > 65536:
        with pytest.raises(AssertionError):
            cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.triton)
    else:
        out_triton, grad_triton = cross_entropy_forward_backward(**kwargs, implementation=CrossEntropyImpl.triton)
        if grad_output is None:
            assert grad_triton is None
        else:
            Assert.rms_close(grad_triton, grad_torch, 5e-3)
        Assert.rms_close(out_triton, out_torch, 5e-3)


@requires_cuda
def test_triton_adam():
    assert TritonConfig.TRITON_ENABLED
    params = torch.randn(4576427, dtype=torch.float32, device="cuda")
    grads = torch.randn_like(params)
    exp_avgs = torch.randn_like(params)
    exp_avg_sqs = torch.randn_like(params).abs()

    out_params = [params]
    out_grads = [grads]
    out_exp_avgs = [exp_avgs]
    out_exp_avg_sqs = [exp_avg_sqs]

    for noop in (0, 1):
        for use_triton in (False, True):
            params1 = params.clone()
            grads1 = grads.clone()
            exp_avgs1 = exp_avgs.clone()
            exp_avg_sqs1 = exp_avg_sqs.clone()

            out_params.append(params1)
            out_grads.append(grads1)
            out_exp_avgs.append(exp_avgs1)
            out_exp_avg_sqs.append(exp_avg_sqs1)

            triton_adam(
                params1,
                grads1,
                exp_avgs1,
                exp_avg_sqs1,
                noop_flag=params.new_full((1,), noop, dtype=torch.int64),
                grad_scale=params.new_full((1,), 8.6),
                lr=0.1,
                beta1=0.6,
                beta2=0.8,
                step=4,
                weight_decay=0.1,
                epsilon=1e-5,
                use_triton=use_triton,
            )

    def compare(i, j, fn, arg):
        fn(rms_diff(out_params[i], out_params[j]), arg)
        fn(rms_diff(out_exp_avgs[i], out_exp_avgs[j]), arg)
        fn(rms_diff(out_exp_avg_sqs[i], out_exp_avg_sqs[j]), arg)

    # Update does something
    compare(0, 1, Assert.geq, 1e-3)
    compare(0, 2, Assert.geq, 1e-3)

    # Updates match
    compare(1, 2, Assert.leq, 1e-3)

    # Noop
    compare(0, 3, Assert.eq, 0)
    compare(0, 4, Assert.eq, 0)


@requires_cuda
@pytest.mark.parametrize(
    ("num_rows_dense", "num_experts", "num_experts_per_token"),
    [(2048, 8, 2), (2048, 6, 2), (2048, 8, 8), (256, 8, 2), (5627, 8, 2)],
)
def test_triton_sparse_map(num_rows_dense, num_experts, num_experts_per_token):
    logits = torch.randn((num_rows_dense, num_experts), device="cuda")
    _, top_experts = torch.topk(logits, num_experts_per_token, dim=-1)

    sparse_map_triton = get_sparse_map(top_experts, num_experts, use_triton=True)
    sparse_map_pytorch = get_sparse_map(top_experts, num_experts, use_triton=False)

    Assert.eq(num_rows_dense, sparse_map_triton.num_rows_dense, sparse_map_pytorch.num_rows_dense)
    Assert.eq(
        num_rows_dense * num_experts_per_token,
        sparse_map_triton.num_rows_unpadded,
        sparse_map_pytorch.num_rows_unpadded,
    )
    Assert.eq(num_experts, sparse_map_triton.num_experts, sparse_map_pytorch.num_experts)
    Assert.eq(num_experts_per_token, sparse_map_triton.num_experts_per_token, sparse_map_pytorch.num_experts_per_token)
    Assert.eq(sparse_map_triton.num_rows, sparse_map_pytorch.num_rows)
    Assert.multiple(sparse_map_triton.num_rows, MAX_DROPLESS_BLOCK_SIZE_ROW)
    Assert.gt(sparse_map_triton.num_rows, sparse_map_triton.num_rows_unpadded)
    Assert.all_equal(sparse_map_triton.expert_ends, sparse_map_pytorch.expert_ends)
    Assert.all_equal(sparse_map_triton.expert_pad_begins, sparse_map_pytorch.expert_pad_begins)
    Assert.all_equal(sparse_map_triton.sparse_rows, sparse_map_pytorch.sparse_rows)
