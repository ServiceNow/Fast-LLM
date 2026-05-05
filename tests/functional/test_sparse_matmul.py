import dataclasses
import functools

import pytest
import torch

from fast_llm.functional.triton.sparse_copy import SparseMap
from fast_llm.functional.triton.sparse_linear import (
    InputSparseLinear,
    OutputSparseLinear,
    dense_matmul,
    input_inner_sparse_matmul,
    input_row_sparse_matmul,
    output_sparse_matmul,
)
from fast_llm.utils import Assert
from tests.utils.utils import requires_triton


@dataclasses.dataclass
class _SparseTestData:
    dense_dim: int
    sparse_dim: int
    expert_ends: tuple[int, ...]
    tokens_per_expert: tuple[int, ...]
    std: float = 0.125

    @functools.cached_property
    def expert_begins(self) -> tuple[int, ...]:
        return (0,) + self.expert_ends[:-1]

    @functools.cached_property
    def expert_pad_begins(self) -> tuple[int, ...]:
        return tuple(
            expert_begin + expert_tokens
            for expert_begin, expert_tokens in zip(self.expert_begins, self.tokens_per_expert, strict=True)
        )

    @functools.cached_property
    def token_dim(self) -> int:
        return self.expert_ends[-1]

    @property
    def sparse_dim_expanded(self) -> int:
        return self.sparse_dim * self.num_experts

    @functools.cached_property
    def num_experts(self) -> int:
        return len(self.expert_begins)

    def get_sparse_map(self, device: torch.device) -> SparseMap:
        return SparseMap(
            num_experts=self.num_experts,
            expert_ends=torch.tensor(self.expert_ends, device=device),
            expert_pad_begins=torch.tensor(self.expert_pad_begins, device=device),
            num_rows=self.expert_ends[-1],
            # Not needed
            sparse_rows=None,
            num_rows_dense=None,
            num_rows_unpadded=None,
            num_experts_per_token=None,
        )

    def normal(self, dim_0: int, dim_1: int, device: torch.device) -> torch.Tensor:
        return torch.normal(0, self.std, (dim_0, dim_1), device=device)


_SPARSE_TEST_DATAS = (
    _SparseTestData(
        dense_dim=384,
        sparse_dim=256,
        expert_ends=(128, 384, 512),
        tokens_per_expert=(78, 256, 54),
    ),
    _SparseTestData(
        dense_dim=256,
        sparse_dim=512,
        expert_ends=(128, 256, 256, 384),
        tokens_per_expert=(52, 125, 0, 97),  # expert 2 has zero real tokens
    ),
    # Single expert — the simplest non-trivial case; also exercises the no-padding path.
    _SparseTestData(
        dense_dim=512,
        sparse_dim=256,
        expert_ends=(256,),
        tokens_per_expert=(200,),
    ),
    # Four experts, fully packed (no padding rows) — exercises the pad_begin == expert_end path.
    # Expert sizes must be multiples of the largest autotune block_size_row (128); otherwise
    # blocks straddle expert boundaries and the kernel's "sparse_index constant within a block"
    # assumption breaks.
    _SparseTestData(
        dense_dim=384,
        sparse_dim=128,
        expert_ends=(128, 256, 384, 512),
        tokens_per_expert=(128, 128, 128, 128),
    ),
)


@requires_triton
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
def test_dense_matmul(sparse_test_data, testing_device):
    lhs = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.dense_dim, testing_device)
    rhs = sparse_test_data.normal(sparse_test_data.dense_dim, sparse_test_data.sparse_dim, testing_device)

    output = dense_matmul(lhs, rhs)
    output_ref = torch.matmul(lhs, rhs)
    Assert.rms_close(output, output_ref, 1e-3)


@requires_triton
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
def test_output_sparse_matmul(sparse_test_data, testing_device):
    lhs = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.dense_dim, testing_device)
    rhs = sparse_test_data.normal(sparse_test_data.dense_dim, sparse_test_data.sparse_dim_expanded, testing_device)

    # Randomly initialize the output to ensure padded values have no effect.
    out = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.sparse_dim, testing_device)
    output = output_sparse_matmul(lhs, rhs, sparse_test_data.get_sparse_map(testing_device), out)

    output_ref = torch.zeros_like(output)
    for i in range(sparse_test_data.num_experts):
        # Padded tokens are treated like regular ones.
        output_ref[sparse_test_data.expert_begins[i] : sparse_test_data.expert_ends[i]] = torch.matmul(
            lhs[sparse_test_data.expert_begins[i] : sparse_test_data.expert_ends[i]],
            rhs[:, i * sparse_test_data.sparse_dim : (i + 1) * sparse_test_data.sparse_dim],
        )

    Assert.rms_close(output, output_ref, 1e-3)


@requires_triton
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
def test_input_inner_sparse_matmul(sparse_test_data, testing_device):
    lhs = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.sparse_dim, testing_device)
    rhs = sparse_test_data.normal(sparse_test_data.sparse_dim_expanded, sparse_test_data.dense_dim, testing_device)

    output = input_inner_sparse_matmul(lhs, rhs, sparse_test_data.get_sparse_map(testing_device))

    output_ref = torch.zeros_like(output)
    for i in range(sparse_test_data.num_experts):
        # Padded tokens are treated like regular ones.
        output_ref[sparse_test_data.expert_begins[i] : sparse_test_data.expert_ends[i]] = torch.matmul(
            lhs[sparse_test_data.expert_begins[i] : sparse_test_data.expert_ends[i]],
            rhs[i * sparse_test_data.sparse_dim : (i + 1) * sparse_test_data.sparse_dim],
        )

    Assert.rms_close(output, output_ref, 1e-3)


@requires_triton
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
def test_input_row_sparse_matmul(sparse_test_data, testing_device):
    lhs = sparse_test_data.normal(sparse_test_data.sparse_dim, sparse_test_data.token_dim, testing_device)
    rhs = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.dense_dim, testing_device)

    output = input_row_sparse_matmul(lhs, rhs, sparse_test_data.get_sparse_map(testing_device))

    output_ref = torch.zeros_like(output)
    for i in range(sparse_test_data.num_experts):
        # Padded tokens are excluded from the sum.
        output_ref[i * sparse_test_data.sparse_dim : (i + 1) * sparse_test_data.sparse_dim] = torch.matmul(
            lhs[:, sparse_test_data.expert_begins[i] : sparse_test_data.expert_pad_begins[i]],
            rhs[sparse_test_data.expert_begins[i] : sparse_test_data.expert_pad_begins[i]],
        )

    Assert.rms_close(output, output_ref, 1e-3)


# --------------------------------------------------------------------------- autograd wrappers


def _sparse_linear_ref(lhs: torch.Tensor, rhs: torch.Tensor, data: _SparseTestData, expert_axis: int) -> torch.Tensor:
    """Per-expert matmul reference; rows past `expert_pad_begins` are zero in the output."""
    rhs_per_expert = rhs.chunk(data.num_experts, dim=expert_axis)
    out = lhs.new_zeros(data.token_dim, rhs_per_expert[0].shape[1])
    for i, (begin, end) in enumerate(zip(data.expert_begins, data.expert_pad_begins, strict=True)):
        out[begin:end] = lhs[begin:end] @ rhs_per_expert[i]
    return out


def _zero_padded_rows(tensor: torch.Tensor, data: _SparseTestData) -> torch.Tensor:
    # The autograd kernels treat padded tokens as regular ones; forward output and grad_lhs
    # contain matmul-of-random garbage in [pad_begin, expert_end). Zero those rows so the
    # comparison vs the reference (which produces zeros there) reflects only real-row error.
    masked = tensor.clone()
    for begin, end in zip(data.expert_pad_begins, data.expert_ends, strict=True):
        if end > begin:
            masked[begin:end] = 0
    return masked


@requires_triton
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
@pytest.mark.parametrize(
    "autograd_class,expert_axis",
    [(OutputSparseLinear, 1), (InputSparseLinear, 0)],
    ids=["output_sparse", "input_sparse"],
)
def test_sparse_linear_autograd(sparse_test_data, testing_device, autograd_class, expert_axis):
    # `expert_axis` is the rhs axis split per expert. Matmul contracts rhs's axis 0, so
    # OutputSparseLinear (expert_axis=1) splits the output dim, InputSparseLinear (expert_axis=0)
    # splits the contracting dim.
    if expert_axis == 1:
        lhs_features, out_features = sparse_test_data.dense_dim, sparse_test_data.sparse_dim
        rhs_shape = (sparse_test_data.dense_dim, sparse_test_data.sparse_dim_expanded)
    else:
        lhs_features, out_features = sparse_test_data.sparse_dim, sparse_test_data.dense_dim
        rhs_shape = (sparse_test_data.sparse_dim_expanded, sparse_test_data.dense_dim)

    lhs = sparse_test_data.normal(sparse_test_data.token_dim, lhs_features, testing_device)
    rhs = sparse_test_data.normal(*rhs_shape, testing_device)
    grad_output = sparse_test_data.normal(sparse_test_data.token_dim, out_features, testing_device)

    lhs_ref = lhs.detach().requires_grad_(True)
    rhs_ref = rhs.detach().requires_grad_(True)
    out_ref = _sparse_linear_ref(lhs_ref, rhs_ref, sparse_test_data, expert_axis)
    out_ref.backward(grad_output)

    lhs_t = lhs.detach().requires_grad_(True)
    rhs_t = rhs.detach().requires_grad_(True)
    out_t = autograd_class.apply(lhs_t, rhs_t, sparse_test_data.get_sparse_map(testing_device))
    out_t.backward(grad_output)

    Assert.rms_close(_zero_padded_rows(out_t, sparse_test_data), out_ref, 1e-3)
    Assert.rms_close(_zero_padded_rows(lhs_t.grad, sparse_test_data), lhs_ref.grad, 1e-3)
    Assert.rms_close(rhs_t.grad, rhs_ref.grad, 1e-3)
