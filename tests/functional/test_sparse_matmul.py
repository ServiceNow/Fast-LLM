import dataclasses
import functools

import pytest
import torch

from fast_llm.functional.triton.sparse_copy import SparseMap
from fast_llm.functional.triton.sparse_linear import (
    dense_matmul,
    input_inner_sparse_matmul,
    input_row_sparse_matmul,
    output_sparse_matmul,
)
from fast_llm.utils import Assert
from tests.utils.utils import requires_cuda


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

    @functools.cached_property
    def sparse_map(self) -> SparseMap:
        return SparseMap(
            num_experts=self.num_experts,
            expert_ends=torch.tensor(self.expert_ends, device="cuda"),
            expert_pad_begins=torch.tensor(self.expert_pad_begins, device="cuda"),
            num_rows=self.expert_ends[-1],
            # Not needed
            sparse_rows=None,
            num_rows_dense=None,
            num_rows_unpadded=None,
            num_experts_per_token=None,
        )

    def normal(self, dim_0: int, dim_1: int) -> torch.Tensor:
        return torch.normal(0, self.std, (dim_0, dim_1), device="cuda")


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
        tokens_per_expert=(52, 125, 0, 97),
    ),
)


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
def test_dense_matmul(sparse_test_data):
    lhs = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.dense_dim)
    rhs = sparse_test_data.normal(sparse_test_data.dense_dim, sparse_test_data.sparse_dim)

    output = dense_matmul(lhs, rhs)
    output_ref = torch.matmul(lhs, rhs)
    Assert.rms_close(output, output_ref, 1e-3)


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
def test_output_sparse_matmul(sparse_test_data):
    lhs = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.dense_dim)
    rhs = sparse_test_data.normal(sparse_test_data.dense_dim, sparse_test_data.sparse_dim_expanded)

    # Randomly initialize the output to ensure padded values have no effect.
    out = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.sparse_dim)
    output = output_sparse_matmul(lhs, rhs, sparse_test_data.sparse_map, out)

    output_ref = torch.zeros_like(output)
    for i in range(sparse_test_data.num_experts):
        # Padded tokens are treated like regular ones.
        output_ref[sparse_test_data.expert_begins[i] : sparse_test_data.expert_ends[i]] = torch.matmul(
            lhs[sparse_test_data.expert_begins[i] : sparse_test_data.expert_ends[i]],
            rhs[:, i * sparse_test_data.sparse_dim : (i + 1) * sparse_test_data.sparse_dim],
        )

    Assert.rms_close(output, output_ref, 1e-3)


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
def test_input_inner_sparse_matmul(sparse_test_data):
    lhs = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.sparse_dim)
    rhs = sparse_test_data.normal(sparse_test_data.sparse_dim_expanded, sparse_test_data.dense_dim)

    output = input_inner_sparse_matmul(lhs, rhs, sparse_test_data.sparse_map)

    output_ref = torch.zeros_like(output)
    for i in range(sparse_test_data.num_experts):
        # Padded tokens are treated like regular ones.
        output_ref[sparse_test_data.expert_begins[i] : sparse_test_data.expert_ends[i]] = torch.matmul(
            lhs[sparse_test_data.expert_begins[i] : sparse_test_data.expert_ends[i]],
            rhs[i * sparse_test_data.sparse_dim : (i + 1) * sparse_test_data.sparse_dim],
        )

    Assert.rms_close(output, output_ref, 1e-3)


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("sparse_test_data", _SPARSE_TEST_DATAS)
def test_input_row_sparse_matmul(sparse_test_data):
    lhs = sparse_test_data.normal(sparse_test_data.sparse_dim, sparse_test_data.token_dim)
    rhs = sparse_test_data.normal(sparse_test_data.token_dim, sparse_test_data.dense_dim)

    output = input_row_sparse_matmul(lhs, rhs, sparse_test_data.sparse_map)

    output_ref = torch.zeros_like(output)
    for i in range(sparse_test_data.num_experts):
        # Padded tokens are excluded from the sum.
        output_ref[i * sparse_test_data.sparse_dim : (i + 1) * sparse_test_data.sparse_dim] = torch.matmul(
            lhs[:, sparse_test_data.expert_begins[i] : sparse_test_data.expert_pad_begins[i]],
            rhs[sparse_test_data.expert_begins[i] : sparse_test_data.expert_pad_begins[i]],
        )

    Assert.rms_close(output, output_ref, 1e-3)
