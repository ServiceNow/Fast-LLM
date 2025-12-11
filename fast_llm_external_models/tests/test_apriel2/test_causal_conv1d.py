"""Tests for CausalConv1d consistency across all code paths.

The Key Consistency Property
============================
For ANY input sequence, ALL of the following must produce the SAME output:

1. Prefill entire sequence at once (CPU/PyTorch fallback)
2. Prefill entire sequence at once (CUDA fast path)
3. Prefill in chunks with state passing (CPU)
4. Prefill in chunks with state passing (CUDA)
5. Prefill prefix + decode remaining tokens one-by-one (CPU)
6. Prefill prefix + decode remaining tokens one-by-one (CUDA)
7. Mixed: CUDA prefill → CPU decode
8. Mixed: CPU prefill → CUDA decode

This is critical because during inference:
- Prefill processes the prompt (potentially chunked for long prompts)
- Decode generates tokens one at a time
- If these paths diverge, generation quality degrades silently
"""

import pytest
import torch

from fast_llm_external_models.apriel2.modeling_apriel2 import CausalConv1d, _causal_conv1d_fn


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def conv():
    """CausalConv1d layer with fixed random weights (on CPU)."""
    torch.manual_seed(42)
    return CausalConv1d(
        in_channels=64,
        out_channels=64,
        kernel_size=4,
        groups=64,
        bias=True,
        activation="silu",
        device="cpu",
    )


@pytest.fixture
def dim():
    return 64


@pytest.fixture
def kernel_size():
    return 4


# =============================================================================
# Helpers
# =============================================================================


def to_device(conv: CausalConv1d, device: str) -> CausalConv1d:
    """Create a copy of conv on the specified device."""
    import copy
    return copy.deepcopy(conv).to(device)


def prefill(conv: CausalConv1d, x: torch.Tensor, state: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill and return (output, final_state)."""
    return conv(x, conv_state=state, return_final_state=True)


def decode_sequence(conv: CausalConv1d, tokens: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode multiple tokens one-by-one, return (stacked_outputs, final_state).

    Args:
        conv: CausalConv1d layer
        tokens: [batch, dim, num_tokens] - tokens to decode
        state: [batch, dim, kernel_size-1] - initial state (modified in-place)

    Returns:
        outputs: [batch, dim, num_tokens] - output for each token
        state: final state after all tokens
    """
    outputs = []
    for i in range(tokens.shape[-1]):
        token = tokens[:, :, i]
        out = conv.update(token, state)
        outputs.append(out)
    return torch.stack(outputs, dim=-1), state


# =============================================================================
# Unit Tests
# =============================================================================


class TestCausalConv1dBasics:
    """Basic functionality tests."""

    def test_output_shape(self, conv, dim):
        """Output shape matches input shape."""
        x = torch.randn(2, dim, 16, device="cpu")
        out = conv(x)
        assert out.shape == x.shape

    def test_state_shape(self, conv, dim, kernel_size):
        """Returned state has correct shape."""
        x = torch.randn(2, dim, 16, device="cpu")
        out, state = conv(x, return_final_state=True)
        assert state.shape == (2, dim, kernel_size - 1)

    def test_deterministic(self, conv, dim):
        """Same input produces same output."""
        x = torch.randn(2, dim, 16, device="cpu")
        out1 = conv(x)
        out2 = conv(x)
        torch.testing.assert_close(out1, out2)

    def test_update_output_shape(self, conv, dim, kernel_size):
        """Update produces single token output."""
        token = torch.randn(2, dim, device="cpu")
        state = torch.randn(2, dim, kernel_size - 1, device="cpu")
        out = conv.update(token, state)
        assert out.shape == (2, dim)

    def test_fast_path_detection(self, conv, dim):
        """Fast path correctly detected based on device."""
        x_cpu = torch.randn(2, dim, 16, device="cpu")
        assert not conv._use_fast_path(x_cpu)

        if torch.cuda.is_available():
            x_cuda = torch.randn(2, dim, 16, device="cuda")
            conv_cuda = conv.cuda()
            # Fast path available only if CUDA kernels installed
            expected = _causal_conv1d_fn is not None
            assert conv_cuda._use_fast_path(x_cuda) == expected


# =============================================================================
# Backend Equivalence (CUDA vs CPU)
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(_causal_conv1d_fn is None, reason="CUDA conv kernels required")
class TestBackendEquivalence:
    """CUDA and CPU backends produce identical results."""

    @pytest.mark.parametrize("seq_len", [1, 4, 8, 17, 32, 65])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_prefill_cuda_vs_cpu(self, conv, dim, seq_len, batch_size):
        """CUDA prefill matches CPU prefill."""
        torch.manual_seed(123)
        x = torch.randn(batch_size, dim, seq_len, device="cpu")

        # CPU
        out_cpu = conv(x)

        # CUDA
        conv_cuda = to_device(conv, "cuda")
        out_cuda = conv_cuda(x.cuda()).cpu()

        torch.testing.assert_close(out_cuda, out_cpu, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("seq_len", [1, 4, 8, 17, 32])
    def test_prefill_with_state_cuda_vs_cpu(self, conv, dim, kernel_size, seq_len):
        """CUDA prefill with state output matches CPU."""
        torch.manual_seed(123)
        x = torch.randn(2, dim, seq_len, device="cpu")

        # CPU
        out_cpu, state_cpu = prefill(conv, x)

        # CUDA
        conv_cuda = to_device(conv, "cuda")
        out_cuda, state_cuda = prefill(conv_cuda, x.cuda())
        out_cuda, state_cuda = out_cuda.cpu(), state_cuda.cpu()

        torch.testing.assert_close(out_cuda, out_cpu, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state_cuda, state_cpu, atol=1e-5, rtol=1e-5)

    def test_decode_cuda_vs_cpu(self, conv, dim, kernel_size):
        """CUDA single-token decode matches CPU."""
        torch.manual_seed(123)
        token = torch.randn(2, dim, device="cpu")
        state = torch.randn(2, dim, kernel_size - 1, device="cpu")

        # CPU
        state_cpu = state.clone()
        out_cpu = conv.update(token, state_cpu)

        # CUDA
        conv_cuda = to_device(conv, "cuda")
        state_cuda = state.cuda()
        out_cuda = conv_cuda.update(token.cuda(), state_cuda).cpu()
        state_cuda = state_cuda.cpu()

        torch.testing.assert_close(out_cuda, out_cpu, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state_cuda, state_cpu, atol=1e-5, rtol=1e-5)


# =============================================================================
# Chunking Consistency
# =============================================================================


class TestChunkingConsistency:
    """Chunked prefill matches full prefill."""

    @pytest.mark.parametrize("total_len", [16, 33, 64])
    @pytest.mark.parametrize("chunk_size", [4, 7, 16])
    def test_chunked_prefill_cpu(self, conv, dim, total_len, chunk_size):
        """CPU: Chunked prefill matches full prefill."""
        torch.manual_seed(123)
        x = torch.randn(2, dim, total_len, device="cpu")

        # Reference: full prefill
        ref_out, _ = prefill(conv, x)

        # Chunked prefill
        outputs = []
        state = None
        for start in range(0, total_len, chunk_size):
            chunk = x[:, :, start:start + chunk_size]
            out, state = prefill(conv, chunk, state)
            outputs.append(out)

        chunked_out = torch.cat(outputs, dim=-1)
        torch.testing.assert_close(chunked_out, ref_out, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(_causal_conv1d_fn is None, reason="CUDA conv kernels required")
    @pytest.mark.parametrize("total_len", [16, 33, 64])
    @pytest.mark.parametrize("chunk_size", [4, 7, 16])
    def test_chunked_prefill_cuda(self, conv, dim, total_len, chunk_size):
        """CUDA: Chunked prefill matches full prefill."""
        torch.manual_seed(123)
        x = torch.randn(2, dim, total_len, device="cpu")

        conv_cuda = to_device(conv, "cuda")

        # Reference: full prefill
        ref_out, _ = prefill(conv_cuda, x.cuda())

        # Chunked prefill
        outputs = []
        state = None
        for start in range(0, total_len, chunk_size):
            chunk = x[:, :, start:start + chunk_size].cuda()
            out, state = prefill(conv_cuda, chunk, state)
            outputs.append(out)

        chunked_out = torch.cat(outputs, dim=-1)
        torch.testing.assert_close(chunked_out, ref_out, atol=1e-4, rtol=1e-4)


# =============================================================================
# Decode Consistency
# =============================================================================


class TestDecodeConsistency:
    """Token-by-token decode matches batch prefill."""

    @pytest.mark.parametrize("prefill_len", [4, 8, 16])
    @pytest.mark.parametrize("decode_len", [1, 5, 10])
    def test_prefill_then_decode_cpu(self, conv, dim, prefill_len, decode_len):
        """CPU: Prefill + decode matches full prefill."""
        torch.manual_seed(123)
        total_len = prefill_len + decode_len
        x = torch.randn(2, dim, total_len, device="cpu")

        # Reference: full prefill
        ref_out, _ = prefill(conv, x)

        # Prefill prefix, then decode rest
        out_prefix, state = prefill(conv, x[:, :, :prefill_len])
        out_decode, _ = decode_sequence(conv, x[:, :, prefill_len:], state)

        combined = torch.cat([out_prefix, out_decode], dim=-1)
        torch.testing.assert_close(combined, ref_out, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(_causal_conv1d_fn is None, reason="CUDA conv kernels required")
    @pytest.mark.parametrize("prefill_len", [4, 8, 16])
    @pytest.mark.parametrize("decode_len", [1, 5, 10])
    def test_prefill_then_decode_cuda(self, conv, dim, prefill_len, decode_len):
        """CUDA: Prefill + decode matches full prefill."""
        torch.manual_seed(123)
        total_len = prefill_len + decode_len
        x = torch.randn(2, dim, total_len, device="cuda")

        conv_cuda = to_device(conv, "cuda")

        # Reference: full prefill
        ref_out, _ = prefill(conv_cuda, x)

        # Prefill prefix, then decode rest
        out_prefix, state = prefill(conv_cuda, x[:, :, :prefill_len])
        out_decode, _ = decode_sequence(conv_cuda, x[:, :, prefill_len:], state)

        combined = torch.cat([out_prefix, out_decode], dim=-1)
        torch.testing.assert_close(combined, ref_out, atol=1e-4, rtol=1e-4)


# =============================================================================
# Global Consistency: The Ultimate Test
# =============================================================================


class TestGlobalConsistency:
    """ALL code paths must produce identical results for the same input."""

    def test_all_cpu_paths_match(self, conv, dim):
        """All CPU paths produce identical output."""
        torch.manual_seed(42)

        total_len = 24
        prefill_len = 16
        chunk_size = 8
        x = torch.randn(2, dim, total_len, device="cpu")

        # Reference: full prefill
        reference, _ = prefill(conv, x)

        # Path 1: Chunked prefill
        outputs = []
        state = None
        for start in range(0, total_len, chunk_size):
            chunk = x[:, :, start:start + chunk_size]
            out, state = prefill(conv, chunk, state)
            outputs.append(out)
        path1 = torch.cat(outputs, dim=-1)

        # Path 2: Prefill + decode
        out_prefix, state = prefill(conv, x[:, :, :prefill_len])
        out_decode, _ = decode_sequence(conv, x[:, :, prefill_len:], state)
        path2 = torch.cat([out_prefix, out_decode], dim=-1)

        # Path 3: All decode (extreme case)
        # Prefill first kernel_size-1 tokens, decode rest
        init_len = conv.kernel_size[0] - 1
        out_init, state = prefill(conv, x[:, :, :init_len])
        out_decode, _ = decode_sequence(conv, x[:, :, init_len:], state)
        path3 = torch.cat([out_init, out_decode], dim=-1)

        torch.testing.assert_close(path1, reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(path2, reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(path3, reference, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(_causal_conv1d_fn is None, reason="CUDA conv kernels required")
    def test_all_paths_match_cross_device(self, conv, dim):
        """All paths (CPU and CUDA) produce identical output."""
        torch.manual_seed(42)

        total_len = 24
        prefill_len = 16
        chunk_size = 8
        x = torch.randn(2, dim, total_len, device="cpu")

        conv_cuda = to_device(conv, "cuda")

        # REFERENCE: CPU full prefill (simplest, most trustworthy)
        reference, _ = prefill(conv, x)

        results = {}

        # CPU paths
        # ---------

        # CPU chunked
        outputs, state = [], None
        for start in range(0, total_len, chunk_size):
            out, state = prefill(conv, x[:, :, start:start + chunk_size], state)
            outputs.append(out)
        results["cpu_chunked"] = torch.cat(outputs, dim=-1)

        # CPU prefill + decode
        out_prefix, state = prefill(conv, x[:, :, :prefill_len])
        out_decode, _ = decode_sequence(conv, x[:, :, prefill_len:], state)
        results["cpu_prefill_decode"] = torch.cat([out_prefix, out_decode], dim=-1)

        # CUDA paths
        # ----------

        # CUDA full prefill
        results["cuda_full"], _ = prefill(conv_cuda, x.cuda())
        results["cuda_full"] = results["cuda_full"].cpu()

        # CUDA chunked
        outputs, state = [], None
        for start in range(0, total_len, chunk_size):
            out, state = prefill(conv_cuda, x[:, :, start:start + chunk_size].cuda(), state)
            outputs.append(out.cpu())
        results["cuda_chunked"] = torch.cat(outputs, dim=-1)

        # CUDA prefill + decode
        out_prefix, state = prefill(conv_cuda, x[:, :, :prefill_len].cuda())
        out_decode, _ = decode_sequence(conv_cuda, x[:, :, prefill_len:].cuda(), state)
        results["cuda_prefill_decode"] = torch.cat([out_prefix.cpu(), out_decode.cpu()], dim=-1)

        # Mixed paths
        # -----------

        # CPU prefill, CUDA decode
        out_prefix, state = prefill(conv, x[:, :, :prefill_len])
        state = state.cuda()
        out_decode, _ = decode_sequence(conv_cuda, x[:, :, prefill_len:].cuda(), state)
        results["cpu_prefill_cuda_decode"] = torch.cat([out_prefix, out_decode.cpu()], dim=-1)

        # CUDA prefill, CPU decode
        out_prefix, state = prefill(conv_cuda, x[:, :, :prefill_len].cuda())
        out_prefix, state = out_prefix.cpu(), state.cpu()
        out_decode, _ = decode_sequence(conv, x[:, :, prefill_len:], state)
        results["cuda_prefill_cpu_decode"] = torch.cat([out_prefix, out_decode], dim=-1)

        # Verify all match reference
        tolerances = {
            "cpu_chunked": 1e-5,
            "cpu_prefill_decode": 1e-5,
            "cuda_full": 1e-4,
            "cuda_chunked": 1e-4,
            "cuda_prefill_decode": 1e-4,
            "cpu_prefill_cuda_decode": 1e-4,
            "cuda_prefill_cpu_decode": 1e-4,
        }

        for name, result in results.items():
            tol = tolerances[name]
            torch.testing.assert_close(
                result, reference, atol=tol, rtol=tol,
                msg=f"Path '{name}' diverged from reference"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(_causal_conv1d_fn is None, reason="CUDA conv kernels required")
    def test_long_decode_no_drift(self, conv, dim):
        """Long decode sequence doesn't accumulate errors."""
        torch.manual_seed(42)

        prefill_len = 8
        decode_len = 100  # Long decode to catch drift
        total_len = prefill_len + decode_len
        x = torch.randn(2, dim, total_len, device="cpu")

        conv_cuda = to_device(conv, "cuda")

        # Reference: CPU full prefill
        reference, _ = prefill(conv, x)

        # CUDA prefill + long decode
        out_prefix, state = prefill(conv_cuda, x[:, :, :prefill_len].cuda())
        out_decode, _ = decode_sequence(conv_cuda, x[:, :, prefill_len:].cuda(), state)
        result = torch.cat([out_prefix.cpu(), out_decode.cpu()], dim=-1)

        # Check max error at each position doesn't grow
        errors = (result - reference).abs().max(dim=1).values.max(dim=0).values  # [seq_len]

        # First positions should have small error
        assert errors[:prefill_len].max() < 1e-4, "Prefill error too large"

        # Decode errors shouldn't grow unboundedly
        # Allow slightly more tolerance for later positions but not exponential growth
        assert errors[prefill_len:].max() < 1e-3, "Decode error too large"

        # Check no systematic drift (errors shouldn't consistently increase)
        decode_errors = errors[prefill_len:]
        first_half = decode_errors[:len(decode_errors)//2].mean()
        second_half = decode_errors[len(decode_errors)//2:].mean()
        assert second_half < first_half * 2, "Errors growing over decode steps (drift detected)"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_token_prefill(self, conv, dim, kernel_size):
        """Prefill with just 1 token works."""
        x = torch.randn(2, dim, 1, device="cpu")
        out, state = prefill(conv, x)

        assert out.shape == (2, dim, 1)
        assert state.shape == (2, dim, kernel_size - 1)

    def test_seq_shorter_than_kernel(self, conv, dim, kernel_size):
        """Sequence shorter than kernel_size works."""
        seq_len = kernel_size - 2  # Shorter than kernel
        x = torch.randn(2, dim, seq_len, device="cpu")
        out, state = prefill(conv, x)

        assert out.shape == (2, dim, seq_len)
        assert state.shape == (2, dim, kernel_size - 1)

    def test_seq_exactly_kernel_size(self, conv, dim, kernel_size):
        """Sequence exactly kernel_size works."""
        x = torch.randn(2, dim, kernel_size, device="cpu")
        out, state = prefill(conv, x)

        assert out.shape == (2, dim, kernel_size)

    def test_batch_size_one(self, conv, dim):
        """Batch size 1 works."""
        x = torch.randn(1, dim, 16, device="cpu")
        out, state = prefill(conv, x)

        assert out.shape == (1, dim, 16)

    def test_empty_decode_after_prefill(self, conv, dim, kernel_size):
        """Zero decode steps after prefill is valid."""
        x = torch.randn(2, dim, 16, device="cpu")
        out_prefill, state = prefill(conv, x)

        # No decode, just verify state is usable
        token = torch.randn(2, dim, device="cpu")
        out_token = conv.update(token, state)
        assert out_token.shape == (2, dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(_causal_conv1d_fn is None, reason="CUDA conv kernels required")
    def test_state_device_transfer(self, conv, dim, kernel_size):
        """State can be transferred between devices."""
        x = torch.randn(2, dim, 16, device="cpu")

        # Prefill on CPU
        _, state_cpu = prefill(conv, x)

        # Transfer state to CUDA
        state_cuda = state_cpu.cuda()
        conv_cuda = to_device(conv, "cuda")

        # Decode on CUDA with transferred state
        token = torch.randn(2, dim, device="cuda")
        out = conv_cuda.update(token, state_cuda)

        assert out.shape == (2, dim)
        assert out.device.type == "cuda"
