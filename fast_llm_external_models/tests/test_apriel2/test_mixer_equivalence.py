"""Tests for numerical equivalence between Apriel2 mixers and reference implementations.

This module verifies that Apriel2's mixer implementations produce outputs numerically
equivalent to their reference implementations (HuggingFace transformers, FLA, etc.).

Test Categories:
================
1. DETERMINISM - Verify same input → same output (no random variation)
2. EQUIVALENCE - Verify Apriel2 output matches reference implementation output
3. FAST/SLOW PATH - Verify CUDA kernels match PyTorch fallback

Test Philosophy:
================
- Equivalence tests use the apriel2/conversion module for weight transformations,
  ensuring we test the same code paths used in production checkpoint conversion.
- Determinism tests use fixed seeds and verify bitwise equality.
- All tests use fp32 by default for numerical precision; bf16 is skipped for
  correctness tests (would be used for performance benchmarks).

Mixer Coverage:
===============
- Attention: vs MistralAttention (causal), vs PixtralAttention (non-causal)
- GatedDeltaNet: vs Qwen3NextGatedDeltaNet
- KimiDeltaAttention: vs FLA KimiDeltaAttention
"""

import pytest
import torch
import torch.nn as nn

from fast_llm_external_models.apriel2.conversion import Concat, ExprPlan, Ref, Slice, W, execute

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture(params=[1, 2, 4])
def batch_size(request):
    """Batch sizes to test. Covers single-sample, small batch, and typical batch."""
    return request.param


@pytest.fixture(params=[1, 16, 64, 128])
def seq_len(request):
    """Sequence lengths to test.

    - 1: Single token decode
    - 16: Very short sequence
    - 64: Typical sequence
    - 128: Longer sequence (approaches chunk boundaries)
    """
    return request.param


@pytest.fixture(params=[False, True])
def use_cache(request):
    """Whether to test with cache (multi-phase) or without (single forward pass)."""
    return request.param


@pytest.fixture(params=[4])
def decode_steps(request):
    """Number of decode steps for cache tests. Single value to limit test explosion."""
    return request.param


@pytest.fixture(params=[256, 512])
def hidden_size(request):
    """Hidden sizes to test. 256 is minimal, 512 exercises larger matrices."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param((8, 8, 32), id="mha-8h-32d"),  # MHA: 8 heads, 8 kv heads, 32 head_dim
        pytest.param((8, 4, 32), id="gqa-8h4kv-32d"),  # GQA: 8 heads, 4 kv heads, 32 head_dim
        pytest.param((8, 2, 64), id="gqa-8h2kv-64d"),  # GQA: 8 heads, 2 kv heads, 64 head_dim
        pytest.param((4, 1, 64), id="mqa-4h1kv-64d"),  # MQA: 4 heads, 1 kv head, 64 head_dim
    ]
)
def attention_config(request):
    """Attention head configurations: (num_heads, num_kv_heads, head_dim).

    Covers:
    - MHA (multi-head attention): heads == kv_heads
    - GQA (grouped query attention): heads > kv_heads
    - MQA (multi-query attention): kv_heads == 1
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param((8, 4, 32, 32), id="8v-4k-32d"),  # 8 value heads, 4 key heads, symmetric dims
        pytest.param((8, 2, 64, 64), id="8v-2k-64d"),  # 8 value heads, 2 key heads, larger dims
        pytest.param((4, 2, 32, 64), id="4v-2k-asym"),  # Asymmetric key/value dims
    ]
)
def gdn_config(request):
    """GDN configurations: (value_heads, key_heads, key_head_dim, value_head_dim)."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param((4, 8), id="4h-8d"),  # 4 heads, 8 head_dim (small)
        pytest.param((8, 16), id="8h-16d"),  # 8 heads, 16 head_dim (medium)
        pytest.param((4, 32), id="4h-32d"),  # 4 heads, 32 head_dim (large head_dim)
    ]
)
def kda_config(request):
    """KDA configurations: (num_heads, head_dim)."""
    return request.param


# =============================================================================
# Test Mode Configuration
# =============================================================================


@pytest.fixture(
    params=[
        "precise",
        # "fast" mode (bf16/sdpa) is intentionally skipped:
        # - These are correctness tests, not performance benchmarks
        # - bf16 has ~3 decimal digits precision, masking real bugs
        # - Small tensor sizes make GPU overhead dominate anyway
        pytest.param("fast", marks=pytest.mark.skip(reason="Correctness tests use fp32")),
    ]
)
def test_mode(request):
    """Test configuration mode: 'precise' (fp32/eager) or 'fast' (bf16/sdpa)."""
    return request.param


@pytest.fixture
def test_dtype(test_mode):
    """Dtype derived from test_mode."""
    return torch.float32 if test_mode == "precise" else torch.bfloat16


@pytest.fixture
def attn_impl(test_mode):
    """Attention implementation derived from test_mode."""
    return "eager" if test_mode == "precise" else "sdpa"


@pytest.fixture
def tolerance(test_mode):
    """Tolerance (rtol, atol) derived from test_mode.

    fp32 uses 2e-4 to accommodate minor kernel differences while catching real bugs.
    bf16 would use 1e-2 due to ~3 decimal digit precision.
    """
    return (2e-4, 2e-4) if test_mode == "precise" else (1e-2, 1e-2)


@pytest.fixture(autouse=True)
def override_dtype_for_test_mode(test_mode):
    """Override default dtype based on test_mode.

    Runs after conftest's set_default_dtype fixture.
    """
    dtype = torch.float32 if test_mode == "precise" else torch.bfloat16
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


# =============================================================================
# Helper Functions
# =============================================================================


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    msg: str = "",
):
    """Assert two tensors are close with detailed error diagnostics.

    Args:
        actual: Tensor from implementation under test
        expected: Tensor from reference implementation
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Context message for failure
    """
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_idx = diff.argmax().item()
        raise AssertionError(
            f"{msg}\n"
            f"  Max diff: {max_diff:.6e} at flat index {max_idx}\n"
            f"  Mean diff: {mean_diff:.6e}\n"
            f"  Tolerance: rtol={rtol}, atol={atol}\n"
            f"  Shapes: actual={actual.shape}, expected={expected.shape}"
        )


def assert_deterministic(out1: torch.Tensor, out2: torch.Tensor, mixer_name: str):
    """Assert two outputs from same input are bitwise identical.

    Args:
        out1: First forward pass output
        out2: Second forward pass output
        mixer_name: Name of mixer for error message
    """
    if not torch.equal(out1, out2):
        diff = (out1 - out2).abs()
        max_diff = diff.max().item()
        num_diff = (diff > 0).sum().item()
        raise AssertionError(
            f"{mixer_name} output is not deterministic!\n"
            f"  {num_diff} elements differ (of {diff.numel()} total)\n"
            f"  Max difference: {max_diff:.6e}"
        )


def extract_module_weights(module: nn.Module) -> dict[W, torch.Tensor]:
    """Extract weights from a module as a dict with W keys for conversion plan."""
    weights = {}
    for name, param in module.named_parameters():
        parts = name.split(".")
        key = W(*parts)
        weights[key] = param.data
    return weights


def load_weights_into_module(module: nn.Module, weights: dict[W, torch.Tensor]):
    """Load weights from conversion plan output into a module."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            parts = name.split(".")
            key = W(*parts)
            if key in weights:
                param.copy_(weights[key])


# =============================================================================
# Conversion Plans (Weight Transformations for Equivalence Tests)
# =============================================================================


def plan_mistral_attention_to_apriel2() -> ExprPlan:
    """MistralAttention -> Apriel2Attention weight mapping.

    Both use identical q_proj/k_proj/v_proj/o_proj naming, so this is identity.
    """
    return ExprPlan(
        mappings={
            W("q_proj", "weight"): Ref(key=W("q_proj", "weight")),
            W("k_proj", "weight"): Ref(key=W("k_proj", "weight")),
            W("v_proj", "weight"): Ref(key=W("v_proj", "weight")),
            W("o_proj", "weight"): Ref(key=W("o_proj", "weight")),
        }
    )


def plan_qwen3next_gdn_to_apriel2(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> ExprPlan:
    """Qwen3NextGatedDeltaNet -> Apriel2GatedDeltaNet weight conversion.

    Qwen3Next uses GROUPED layout: for each key_head group, [Q_g | K_g | V_group | Z_group]
    Apriel2/Fast-LLM uses FLAT layout: [Q_all | K_all | V_all | Z_all]

    This plan rearranges in_proj_qkvz weights from grouped to flat layout.
    """
    # Dimensions per group
    v_per_group = (num_v_heads // num_k_heads) * head_v_dim
    group_size = head_k_dim * 2 + v_per_group * 2  # Q + K + V_group + Z_group

    qkvz_ref = Ref(key=W("in_proj_qkvz", "weight"))

    # Extract Q, K, V, Z from each group
    q_slices, k_slices, v_slices, z_slices = [], [], [], []
    for g in range(num_k_heads):
        base = g * group_size
        q_slices.append(Slice(expr=qkvz_ref, slices=((base, base + head_k_dim, None), (None, None, None))))
        k_slices.append(
            Slice(expr=qkvz_ref, slices=((base + head_k_dim, base + 2 * head_k_dim, None), (None, None, None)))
        )
        v_slices.append(
            Slice(
                expr=qkvz_ref,
                slices=((base + 2 * head_k_dim, base + 2 * head_k_dim + v_per_group, None), (None, None, None)),
            )
        )
        z_slices.append(
            Slice(
                expr=qkvz_ref,
                slices=((base + 2 * head_k_dim + v_per_group, base + group_size, None), (None, None, None)),
            )
        )

    in_proj_qkvz_expr = Concat(
        exprs=(
            Concat(exprs=tuple(q_slices), dim=0),
            Concat(exprs=tuple(k_slices), dim=0),
            Concat(exprs=tuple(v_slices), dim=0),
            Concat(exprs=tuple(z_slices), dim=0),
        ),
        dim=0,
    )

    # Similarly rearrange in_proj_ba
    ba_ref = Ref(key=W("in_proj_ba", "weight"))
    ba_per_group = (num_v_heads // num_k_heads) * 2

    b_slices, a_slices = [], []
    for g in range(num_k_heads):
        base = g * ba_per_group
        b_slices.append(
            Slice(expr=ba_ref, slices=((base, base + num_v_heads // num_k_heads, None), (None, None, None)))
        )
        a_slices.append(
            Slice(
                expr=ba_ref,
                slices=((base + num_v_heads // num_k_heads, base + ba_per_group, None), (None, None, None)),
            )
        )

    in_proj_ba_expr = Concat(
        exprs=(Concat(exprs=tuple(b_slices), dim=0), Concat(exprs=tuple(a_slices), dim=0)),
        dim=0,
    )

    return ExprPlan(
        mappings={
            W("in_proj_qkvz", "weight"): in_proj_qkvz_expr,
            W("in_proj_ba", "weight"): in_proj_ba_expr,
            W("out_proj", "weight"): Ref(key=W("out_proj", "weight")),
            W("convolution", "weight"): Ref(key=W("conv1d", "weight")),
            W("dt_bias"): Ref(key=W("dt_bias")),
            W("A_log"): Ref(key=W("A_log")),
            W("norm", "weight"): Ref(key=W("norm", "weight")),
        }
    )


def plan_fla_kda_to_apriel2() -> ExprPlan:
    """FLA KimiDeltaAttention -> Apriel2 KimiDeltaAttention weight mapping.

    Key renames:
    - q_conv1d -> q_conv (same for k, v)
    - f_proj.0/1 -> f_a_proj/f_b_proj
    - g_proj.0/1 -> g_a_proj/g_b_proj
    - b_proj -> beta_proj
    - o_norm -> norm

    Note: FLA has bias on g_proj.1, Apriel2 doesn't. Test zeroes this bias.
    """
    return ExprPlan(
        mappings={
            # Projections (same names)
            W("q_proj", "weight"): Ref(key=W("q_proj", "weight")),
            W("k_proj", "weight"): Ref(key=W("k_proj", "weight")),
            W("v_proj", "weight"): Ref(key=W("v_proj", "weight")),
            W("o_proj", "weight"): Ref(key=W("o_proj", "weight")),
            # Convolutions (conv1d -> conv)
            W("q_conv", "weight"): Ref(key=W("q_conv1d", "weight")),
            W("k_conv", "weight"): Ref(key=W("k_conv1d", "weight")),
            W("v_conv", "weight"): Ref(key=W("v_conv1d", "weight")),
            # Gate projections (Sequential -> separate)
            W("f_a_proj", "weight"): Ref(key=W("f_proj", "0", "weight")),
            W("f_b_proj", "weight"): Ref(key=W("f_proj", "1", "weight")),
            W("g_a_proj", "weight"): Ref(key=W("g_proj", "0", "weight")),
            W("g_b_proj", "weight"): Ref(key=W("g_proj", "1", "weight")),
            # Beta (b_proj -> beta_proj)
            W("beta_proj", "weight"): Ref(key=W("b_proj", "weight")),
            # Learnable params
            W("A_log"): Ref(key=W("A_log")),
            W("dt_bias"): Ref(key=W("dt_bias")),
            # Normalization (o_norm -> norm)
            W("norm", "weight"): Ref(key=W("o_norm", "weight")),
        }
    )


# =============================================================================
# SECTION 1: DETERMINISM TESTS
# =============================================================================


class TestDeterminism:
    """Verify mixers produce deterministic outputs.

    These tests run the same input through a mixer twice and verify
    bitwise-identical outputs. Non-determinism would indicate:
    - Uncontrolled randomness in kernels
    - Race conditions in parallel operations
    - Floating-point non-associativity issues
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_attention_determinism(self, attention_config):
        """Verify Apriel2Attention produces identical output on repeated calls."""
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2Attention

        num_heads, num_kv_heads, head_dim = attention_config
        hidden_size = 256
        batch_size, seq_len = 2, 32

        mixer_config = {
            "type": "attention",
            "heads": num_heads,
            "head_groups": num_kv_heads,
            "head_size": head_dim,
            "add_linear_biases": False,
            "causal": True,
            "rotary": {"type": "mistral_1d", "theta": 10000.0},
        }

        config = Apriel2TextConfig(
            hidden_size=hidden_size,
            decoder={
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": mixer_config,
                    "mlp": {"type": "mlp", "intermediate_size": hidden_size * 4},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
            embeddings={"max_position_embeddings": 4096},
        )
        config._attn_implementation = "eager"

        torch.manual_seed(42)
        model = Apriel2Attention(hidden_size, mixer_config, layer_idx=0, config=config)
        model.eval()

        torch.manual_seed(123)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        rotary_resources = Apriel2Attention.setup(mixer_config, hidden_size, 4096)
        position_embeddings = rotary_resources["rotary_emb"](hidden_states, position_ids)

        with torch.no_grad():
            out1 = model(hidden_states, position_embeddings=position_embeddings)[0]
            out2 = model(hidden_states, position_embeddings=position_embeddings)[0]

        assert_deterministic(out1, out2, "Apriel2Attention")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN requires CUDA")
    def test_gdn_determinism(self, gdn_config):
        """Verify Apriel2GatedDeltaNet produces identical output on repeated calls."""
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet

        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config
        hidden_size = 256
        batch_size, seq_len = 2, 32

        config_dict = {
            "type": "gdn",
            "value_heads": value_heads,
            "key_heads": key_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "convolution_layer": {"kernel_size": 4},
            "norm_eps": 1e-5,
        }

        torch.manual_seed(42)
        model = Apriel2GatedDeltaNet(hidden_size, config_dict, layer_idx=0)
        model.eval()

        torch.manual_seed(123)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            out1 = model(hidden_states)[0]
            out2 = model(hidden_states)[0]

        assert_deterministic(out1, out2, "Apriel2GatedDeltaNet")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA requires CUDA")
    def test_kda_determinism(self, kda_config):
        """Verify Apriel2 KimiDeltaAttention produces identical output on repeated calls."""
        from fast_llm_external_models.apriel2.modeling_apriel2 import KimiDeltaAttention

        num_heads, head_dim = kda_config
        hidden_size = num_heads * head_dim
        batch_size, seq_len = 2, 32

        config_dict = {
            "type": "kda",
            "heads": num_heads,
            "head_dim": head_dim,
            "convolution_layer": {"kernel_size": 4},
            "normalization": {"epsilon": 1e-5},
        }

        torch.manual_seed(42)
        model = KimiDeltaAttention(hidden_size, config_dict, layer_idx=0)
        model.eval()

        torch.manual_seed(123)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        with torch.no_grad():
            out1 = model(hidden_states)[0]
            out2 = model(hidden_states)[0]

        assert_deterministic(out1, out2, "KimiDeltaAttention")


# =============================================================================
# SECTION 2: EQUIVALENCE TESTS - Attention
# =============================================================================


class TestAttentionEquivalence:
    """Verify Apriel2Attention matches reference attention implementations.

    Tests both causal (vs Mistral) and non-causal (vs Pixtral) modes.
    """

    @pytest.fixture
    def mistral_config(self, hidden_size, attention_config, attn_impl):
        """Create MistralConfig for causal attention testing."""
        from transformers import MistralConfig

        num_heads, num_kv_heads, head_dim = attention_config
        config = MistralConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = attn_impl
        return config

    @pytest.fixture
    def apriel2_config(self, hidden_size, attention_config, attn_impl):
        """Create Apriel2Config for causal attention testing."""
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig

        num_heads, num_kv_heads, head_dim = attention_config
        config = Apriel2TextConfig(
            hidden_size=hidden_size,
            decoder={
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "attention",
                        "heads": num_heads,
                        "head_groups": num_kv_heads,
                        "head_size": head_dim,
                        "add_linear_biases": False,
                        "causal": True,
                        "rotary": {"type": "mistral_1d", "theta": 10000.0},
                    },
                    "mlp": {"type": "mlp", "intermediate_size": hidden_size * 4},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
            embeddings={"max_position_embeddings": 4096},
        )
        config._attn_implementation = attn_impl
        return config

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_causal_vs_mistral(
        self,
        mistral_config,
        apriel2_config,
        batch_size,
        seq_len,
        hidden_size,
        tolerance,
    ):
        """Verify Apriel2Attention (causal) matches MistralAttention output."""
        from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRotaryEmbedding

        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2Attention

        mixer_config = apriel2_config.decoder["block"]["mixer"]

        # Create models
        mistral_attn = MistralAttention(mistral_config, layer_idx=0)
        apriel2_attn = Apriel2Attention(hidden_size, mixer_config, layer_idx=0, config=apriel2_config)

        # Transfer weights using conversion plan
        plan = plan_mistral_attention_to_apriel2()
        source_weights = extract_module_weights(mistral_attn)
        target_weights = execute(plan, source_weights, seed=42)
        load_weights_into_module(apriel2_attn, target_weights)

        # Create inputs
        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)

        # Compute rotary embeddings
        mistral_rotary = MistralRotaryEmbedding(config=mistral_config)
        position_embeddings = mistral_rotary(hidden_states, position_ids)

        mistral_attn.eval()
        apriel2_attn.eval()

        with torch.no_grad():
            mistral_out = mistral_attn(
                hidden_states, position_embeddings=position_embeddings, attention_mask=causal_mask
            )[0]
            apriel2_out = apriel2_attn(
                hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings
            )[0]

        rtol, atol = tolerance
        assert_close(
            apriel2_out,
            mistral_out,
            rtol=rtol,
            atol=atol,
            msg=f"Apriel2Attention vs MistralAttention (batch={batch_size}, seq={seq_len}, hidden={hidden_size})",
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    @pytest.mark.parametrize("seq_len", [16, 64])  # Must be perfect squares for 2D position
    def test_noncausal_vs_pixtral(
        self,
        attention_config,
        batch_size,
        seq_len,
        attn_impl,
        tolerance,
    ):
        """Verify Apriel2Attention (non-causal) matches PixtralAttention output."""
        from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
        from transformers.models.pixtral.modeling_pixtral import PixtralAttention, PixtralRotaryEmbedding

        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2Attention

        num_heads, _, head_dim = attention_config
        hidden_size = num_heads * head_dim

        # Verify seq_len is perfect square
        grid_size = int(seq_len**0.5)
        if grid_size * grid_size != seq_len:
            pytest.skip(f"seq_len {seq_len} is not a perfect square for 2D position test")

        # Create configs
        pixtral_config = PixtralVisionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=1,
            rope_theta=10000.0,
        )
        pixtral_config._attn_implementation = attn_impl

        mixer_config = {
            "type": "attention",
            "heads": num_heads,
            "head_groups": num_heads,  # Pixtral uses MHA
            "head_size": head_dim,
            "add_linear_biases": False,
            "causal": False,
            "rotary": {"type": "pixtral_2d", "theta": 10000.0, "patch_size": 16, "max_image_size": 1024},
        }

        apriel2_config = Apriel2TextConfig(
            hidden_size=hidden_size,
            decoder={
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": mixer_config,
                    "mlp": {"type": "mlp", "intermediate_size": hidden_size * 4},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
            embeddings={"max_position_embeddings": 4096},
        )
        apriel2_config._attn_implementation = attn_impl

        # Create models
        pixtral_attn = PixtralAttention(pixtral_config)
        apriel2_attn = Apriel2Attention(hidden_size, mixer_config, layer_idx=0, config=apriel2_config)

        # Transfer weights
        plan = plan_mistral_attention_to_apriel2()
        source_weights = extract_module_weights(pixtral_attn)
        target_weights = execute(plan, source_weights, seed=42)
        load_weights_into_module(apriel2_attn, target_weights)

        # Create inputs
        torch.manual_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        rotary_emb = PixtralRotaryEmbedding(config=pixtral_config)
        position_ids = torch.arange(seq_len)
        cos, sin = rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos.unsqueeze(0), sin.unsqueeze(0))

        pixtral_attn.eval()
        apriel2_attn.eval()

        with torch.no_grad():
            pixtral_out = pixtral_attn(hidden_states, attention_mask=None, position_embeddings=position_embeddings)[0]
            apriel2_out = apriel2_attn(hidden_states, attention_mask=None, position_embeddings=position_embeddings)[0]

        rtol, atol = tolerance
        assert_close(
            apriel2_out,
            pixtral_out,
            rtol=rtol,
            atol=atol,
            msg=f"Apriel2Attention (non-causal) vs PixtralAttention (batch={batch_size}, seq={seq_len})",
        )


# =============================================================================
# SECTION 2: EQUIVALENCE TESTS - GatedDeltaNet
# =============================================================================


class TestGDNEquivalence:
    """Verify Apriel2GatedDeltaNet matches Qwen3NextGatedDeltaNet."""

    @pytest.fixture
    def qwen3_config(self, hidden_size, gdn_config):
        """Create Qwen3NextConfig for GDN testing."""
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config
        return Qwen3NextConfig(
            hidden_size=hidden_size,
            linear_num_value_heads=value_heads,
            linear_num_key_heads=key_heads,
            linear_key_head_dim=key_head_dim,
            linear_value_head_dim=value_head_dim,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            torch_dtype=torch.get_default_dtype(),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN requires CUDA")
    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_vs_qwen3next(
        self,
        qwen3_config,
        gdn_config,
        hidden_size,
        batch_size,
        seq_len,
        seed,
        use_cache,
        decode_steps,
        tolerance,
    ):
        """Verify Apriel2GatedDeltaNet matches Qwen3NextGatedDeltaNet output.

        When use_cache=False: Single forward pass on full sequence.
        When use_cache=True: Three-phase test (prefill → decode → prefill) on same total length.

        Note: Phase 3 with cache diverges because Qwen3Next has a bug where chunk mode
        always uses initial_state=None, ignoring cached recurrent state.
        """
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextDynamicCache,
            Qwen3NextGatedDeltaNet,
        )

        from fast_llm_external_models.apriel2.cache import Apriel2Cache
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet

        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config

        # Skip cache tests when seq_len is too small for 3 phases
        if use_cache and seq_len < decode_steps + 2:
            pytest.skip(f"seq_len={seq_len} too small for cache test with decode_steps={decode_steps}")

        # For cache mode, create config with layer_types (required by Qwen3NextDynamicCache)
        if use_cache:
            qwen3_config = Qwen3NextConfig(
                hidden_size=hidden_size,
                linear_num_value_heads=value_heads,
                linear_num_key_heads=key_heads,
                linear_key_head_dim=key_head_dim,
                linear_value_head_dim=value_head_dim,
                linear_conv_kernel_dim=4,
                rms_norm_eps=1e-5,
                max_position_embeddings=4096,
                num_attention_heads=8,
                num_key_value_heads=2,
                head_dim=64,
                torch_dtype=torch.get_default_dtype(),
                num_hidden_layers=1,
                layer_types=["linear_attention"],
            )

        mixer_config = {
            "type": "gdn",
            "value_heads": value_heads,
            "key_heads": key_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "convolution_layer": {"kernel_size": 4},
            "norm_eps": 1e-5,
        }

        # Create models with same weights
        torch.manual_seed(seed)
        qwen_gdn = Qwen3NextGatedDeltaNet(qwen3_config, layer_idx=0).cuda()
        apriel_gdn = Apriel2GatedDeltaNet(hidden_size, mixer_config, layer_idx=0).cuda()

        # Transfer weights using conversion plan
        plan = plan_qwen3next_gdn_to_apriel2(
            num_k_heads=key_heads,
            num_v_heads=value_heads,
            head_k_dim=key_head_dim,
            head_v_dim=value_head_dim,
        )
        source_weights = extract_module_weights(qwen_gdn)
        target_weights = execute(plan, source_weights, seed=seed)
        load_weights_into_module(apriel_gdn, target_weights)

        qwen_gdn.eval()
        apriel_gdn.eval()

        rtol, atol = tolerance

        # Create full input sequence
        torch.manual_seed(seed + 1)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        if not use_cache:
            # === No cache: single forward pass ===
            with torch.no_grad():
                qwen_out = qwen_gdn(hidden_states)
                apriel_out = apriel_gdn(hidden_states)[0]

            assert_close(
                apriel_out,
                qwen_out,
                rtol=rtol,
                atol=atol,
                msg=f"GDN vs Qwen3Next (batch={batch_size}, seq={seq_len}, cache=False)",
            )
        else:
            # === With cache: three-phase test ===
            # Split sequence: prefill + decode + prefill2 = seq_len
            prefill_len = (seq_len - decode_steps) * 2 // 3
            prefill_len = max(1, prefill_len)  # At least 1 token
            prefill2_len = seq_len - prefill_len - decode_steps
            prefill2_len = max(1, prefill2_len)  # At least 1 token

            # Create caches
            qwen_cache = Qwen3NextDynamicCache(qwen3_config)

            apriel_config = Apriel2TextConfig(
                hidden_size=hidden_size,
                decoder={
                    "type": "fixed",
                    "num_blocks": 1,
                    "block": {"mixer": mixer_config},
                },
            )
            apriel_cache = Apriel2Cache(apriel_config)

            # ========== PHASE 1: Initial Prefill ==========
            prefill_input = hidden_states[:, :prefill_len, :]

            with torch.no_grad():
                qwen_out1 = qwen_gdn(
                    prefill_input,
                    cache_params=qwen_cache,
                    cache_position=torch.arange(prefill_len, device="cuda"),
                )
                apriel_out1 = apriel_gdn(
                    prefill_input,
                    past_key_values=apriel_cache,
                    cache_position=torch.arange(prefill_len, device="cuda"),
                )[0]

            assert_close(
                apriel_out1,
                qwen_out1,
                rtol=rtol,
                atol=atol,
                msg=f"Phase 1 (prefill): output mismatch (batch={batch_size}, prefill={prefill_len})",
            )

            # Compare recurrent states
            assert_close(
                apriel_cache.recurrent_states[0],
                qwen_cache.recurrent_states[0],
                rtol=rtol,
                atol=atol,
                msg="Phase 1: recurrent_state mismatch",
            )

            # ========== PHASE 2: Decode (single tokens) ==========
            for i in range(decode_steps):
                pos = prefill_len + i
                decode_input = hidden_states[:, pos : pos + 1, :]

                with torch.no_grad():
                    qwen_out = qwen_gdn(
                        decode_input,
                        cache_params=qwen_cache,
                        cache_position=torch.tensor([pos], device="cuda"),
                    )
                    apriel_out = apriel_gdn(
                        decode_input,
                        past_key_values=apriel_cache,
                        cache_position=torch.tensor([pos], device="cuda"),
                    )[0]

                assert_close(
                    apriel_out,
                    qwen_out,
                    rtol=rtol,
                    atol=atol,
                    msg=f"Phase 2 (decode step {i}): output mismatch",
                )

            # Compare recurrent states after decode
            assert_close(
                apriel_cache.recurrent_states[0],
                qwen_cache.recurrent_states[0],
                rtol=rtol,
                atol=atol,
                msg="Phase 2: recurrent_state mismatch",
            )

            # ========== PHASE 3: Prefill again (decode→prefill transition) ==========
            # NOTE: Qwen3Next passes initial_state=None in chunk mode, so outputs diverge.
            prefill2_start = prefill_len + decode_steps
            prefill2_input = hidden_states[:, prefill2_start : prefill2_start + prefill2_len, :]

            with torch.no_grad():
                qwen_out3 = qwen_gdn(
                    prefill2_input,
                    cache_params=qwen_cache,
                    cache_position=torch.arange(prefill2_start, prefill2_start + prefill2_len, device="cuda"),
                )
                apriel_out3 = apriel_gdn(
                    prefill2_input,
                    past_key_values=apriel_cache,
                    cache_position=torch.arange(prefill2_start, prefill2_start + prefill2_len, device="cuda"),
                )[0]

            # Phase 3 diverges due to Qwen3Next bug - just verify we can run it
            _ = (qwen_out3, apriel_out3)  # Outputs computed but not compared

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GDN requires CUDA")
    @pytest.mark.parametrize("seed", [42, 123, 456])
    @pytest.mark.parametrize("prefill_len", [4, 8, 16])
    def test_chunked_vs_recurrent(
        self,
        gdn_config,
        seed,
        prefill_len,
    ):
        """Verify GDN recurrent mode (decode) matches chunked mode (prefill).

        This tests the inference path: after prefilling N tokens with chunked mode,
        subsequent single-token decodes using recurrent mode should produce the same
        output as if we had run the full sequence through chunked mode.
        """
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2GatedDeltaNet

        value_heads, key_heads, key_head_dim, value_head_dim = gdn_config
        hidden_size = 256
        batch_size = 2
        total_len = prefill_len + 4  # Prefill + 4 decode steps

        config_dict = {
            "type": "gdn",
            "value_heads": value_heads,
            "key_heads": key_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "convolution_layer": {"kernel_size": 4},
            "norm_eps": 1e-5,
        }

        # Create model
        torch.manual_seed(seed)
        model = Apriel2GatedDeltaNet(hidden_size, config_dict, layer_idx=0)
        model = model.cuda()
        model.eval()

        # Create input sequence
        torch.manual_seed(seed + 1)
        full_hidden_states = torch.randn(batch_size, total_len, hidden_size, device="cuda")

        # === Reference: Run full sequence through chunked mode ===
        with torch.no_grad():
            reference_output = model(full_hidden_states)[0]

        # === Test: Prefill + decode ===
        # Create a simple cache object to hold conv and recurrent states
        class SimpleCache:
            def __init__(self):
                self.conv_states = {0: None}
                self.recurrent_states = {0: None}

        cache = SimpleCache()

        # Prefill phase
        prefill_input = full_hidden_states[:, :prefill_len, :]
        with torch.no_grad():
            prefill_output = model(
                prefill_input,
                past_key_values=cache,
                cache_position=torch.arange(prefill_len, device="cuda"),
            )[0]

        # Decode phase - one token at a time
        decode_outputs = []
        for i in range(prefill_len, total_len):
            decode_input = full_hidden_states[:, i : i + 1, :]
            with torch.no_grad():
                decode_output = model(
                    decode_input,
                    past_key_values=cache,
                    cache_position=torch.tensor([i], device="cuda"),
                )[0]
            decode_outputs.append(decode_output)

        # Concatenate prefill + decode outputs
        test_output = torch.cat([prefill_output] + decode_outputs, dim=1)

        # Use looser tolerance for chunked vs recurrent comparison
        # (different processing order leads to numerical differences)
        assert_close(
            test_output,
            reference_output,
            rtol=1e-3,
            atol=1e-3,
            msg=f"GDN chunked vs recurrent mode (prefill={prefill_len}, total={total_len})",
        )

# =============================================================================
# SECTION 2: EQUIVALENCE TESTS - KimiDeltaAttention
# =============================================================================


class TestKDAEquivalence:
    """Verify Apriel2 KimiDeltaAttention matches FLA KimiDeltaAttention."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA requires CUDA")
    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_vs_fla(
        self,
        kda_config,
        batch_size,
        seq_len,
        seed,
        use_cache,
        decode_steps,
        tolerance,
    ):
        """Verify Apriel2 KimiDeltaAttention matches FLA KimiDeltaAttention output.

        When use_cache=False: Single forward pass on full sequence.
        When use_cache=True: Three-phase test (prefill → decode → prefill) on same total length.

        Unlike GDN (where Qwen3Next has a bug), FLA KDA correctly passes initial_state
        in chunk mode, so all three phases should match.
        """
        from fla.layers.kda import KimiDeltaAttention as FLA_KDA
        from fla.models.utils import Cache as FLACache

        from fast_llm_external_models.apriel2.cache import Apriel2Cache
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2TextConfig
        from fast_llm_external_models.apriel2.modeling_apriel2 import KimiDeltaAttention as Apriel2_KDA

        num_heads, head_dim = kda_config
        hidden_size = num_heads * head_dim

        # Skip cache tests when seq_len is too small for 3 phases
        if use_cache and seq_len < decode_steps + 2:
            pytest.skip(f"seq_len={seq_len} too small for cache test with decode_steps={decode_steps}")

        mixer_config = {
            "type": "kda",
            "heads": num_heads,
            "head_dim": head_dim,
            "convolution_layer": {"kernel_size": 4},
            "normalization": {"epsilon": 1e-5},
        }

        # Create FLA KDA with same weights
        torch.manual_seed(seed)
        fla_kda = FLA_KDA(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_size=4,
            conv_bias=False,
            norm_eps=1e-5,
            layer_idx=0,
        ).cuda()
        # FLA has g_proj.1 bias=True but Apriel2/upstream Kimi doesn't - zero it out
        fla_kda.g_proj[1].bias.data.zero_()

        # Create Apriel2 KDA
        apriel_kda = Apriel2_KDA(hidden_size, mixer_config, layer_idx=0).cuda()

        # Transfer weights using conversion plan
        plan = plan_fla_kda_to_apriel2()
        source_weights = extract_module_weights(fla_kda)
        target_weights = execute(plan, source_weights, seed=seed)
        load_weights_into_module(apriel_kda, target_weights)

        fla_kda.eval()
        apriel_kda.eval()

        rtol, atol = tolerance

        # Create full input sequence
        torch.manual_seed(seed + 1)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        if not use_cache:
            # === No cache: single forward pass ===
            with torch.no_grad():
                # use_cache=True ensures FLA initializes conv cache for short sequences
                fla_out = fla_kda(hidden_states, use_cache=True)[0]
                apriel_out = apriel_kda(hidden_states)[0]

            assert_close(
                apriel_out,
                fla_out,
                rtol=rtol,
                atol=atol,
                msg=f"KDA vs FLA (batch={batch_size}, seq={seq_len}, cache=False)",
            )
        else:
            # === With cache: three-phase test ===
            # Split sequence: prefill + decode + prefill2 = seq_len
            prefill_len = (seq_len - decode_steps) * 2 // 3
            prefill_len = max(1, prefill_len)  # At least 1 token
            prefill2_len = seq_len - prefill_len - decode_steps
            prefill2_len = max(1, prefill2_len)  # At least 1 token

            # Create caches
            fla_cache = FLACache()

            apriel_config = Apriel2TextConfig(
                hidden_size=hidden_size,
                decoder={
                    "type": "fixed",
                    "num_blocks": 1,
                    "block": {"mixer": mixer_config},
                },
            )
            apriel_cache = Apriel2Cache(apriel_config)

            # Force chunk mode for prefill
            fla_kda.mode = "chunk"
            apriel_kda.mode = "chunk"

            # ========== PHASE 1: Initial Prefill ==========
            prefill_input = hidden_states[:, :prefill_len, :]

            with torch.no_grad():
                fla_out1 = fla_kda(
                    prefill_input,
                    past_key_values=fla_cache,
                    use_cache=True,
                )[0]
                apriel_out1 = apriel_kda(
                    prefill_input,
                    past_key_values=apriel_cache,
                )[0]

            assert_close(
                apriel_out1,
                fla_out1,
                rtol=rtol,
                atol=atol,
                msg=f"Phase 1 (prefill): output mismatch (batch={batch_size}, prefill={prefill_len})",
            )

            # Compare recurrent states
            assert_close(
                apriel_cache.recurrent_states[0],
                fla_cache[0]["recurrent_state"],
                rtol=rtol,
                atol=atol,
                msg="Phase 1: recurrent_state mismatch",
            )

            # ========== PHASE 2: Decode (single tokens) ==========
            fla_kda.mode = "fused_recurrent"
            apriel_kda.mode = "fused_recurrent"

            for i in range(decode_steps):
                pos = prefill_len + i
                decode_input = hidden_states[:, pos : pos + 1, :]

                with torch.no_grad():
                    fla_out = fla_kda(
                        decode_input,
                        past_key_values=fla_cache,
                        use_cache=True,
                    )[0]
                    apriel_out = apriel_kda(
                        decode_input,
                        past_key_values=apriel_cache,
                    )[0]

                assert_close(
                    apriel_out,
                    fla_out,
                    rtol=rtol,
                    atol=atol,
                    msg=f"Phase 2 (decode step {i}): output mismatch",
                )

            # Compare recurrent states after decode
            assert_close(
                apriel_cache.recurrent_states[0],
                fla_cache[0]["recurrent_state"],
                rtol=rtol,
                atol=atol,
                msg="Phase 2: recurrent_state mismatch",
            )

            # ========== PHASE 3: Prefill again (decode→prefill transition) ==========
            # FLA KDA correctly uses initial_state in chunk mode, so this should match
            fla_kda.mode = "chunk"
            apriel_kda.mode = "chunk"

            prefill2_start = prefill_len + decode_steps
            prefill2_input = hidden_states[:, prefill2_start : prefill2_start + prefill2_len, :]

            with torch.no_grad():
                fla_out3 = fla_kda(
                    prefill2_input,
                    past_key_values=fla_cache,
                    use_cache=True,
                )[0]
                apriel_out3 = apriel_kda(
                    prefill2_input,
                    past_key_values=apriel_cache,
                )[0]

            assert_close(
                apriel_out3,
                fla_out3,
                rtol=rtol,
                atol=atol,
                msg="Phase 3 (decode→prefill): output mismatch",
            )

            # Compare final recurrent states
            assert_close(
                apriel_cache.recurrent_states[0],
                fla_cache[0]["recurrent_state"],
                rtol=rtol,
                atol=atol,
                msg="Phase 3: recurrent_state mismatch",
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA requires CUDA")
    @pytest.mark.parametrize("seed", [42, 123, 456])
    @pytest.mark.parametrize("prefill_len", [4, 8, 16])
    def test_chunked_vs_recurrent(
        self,
        kda_config,
        seed,
        prefill_len,
    ):
        """Verify KDA recurrent mode (fused_recurrent_kda) matches chunked mode (chunk_kda).

        This tests the inference path: after prefilling N tokens with chunked mode,
        subsequent single-token decodes using recurrent mode should produce the same
        output as if we had run the full sequence through chunked mode.
        """
        from fast_llm_external_models.apriel2.modeling_apriel2 import KimiDeltaAttention

        num_heads, head_dim = kda_config
        hidden_size = num_heads * head_dim
        batch_size = 2
        total_len = prefill_len + 4  # Prefill + 4 decode steps

        config_dict = {
            "type": "kda",
            "heads": num_heads,
            "head_dim": head_dim,
            "convolution_layer": {"kernel_size": 4},
            "normalization": {"epsilon": 1e-5},
        }

        # Create model
        torch.manual_seed(seed)
        model = KimiDeltaAttention(hidden_size, config_dict, layer_idx=0)
        model = model.cuda()
        model.eval()

        # Create input sequence
        torch.manual_seed(seed + 1)
        full_hidden_states = torch.randn(batch_size, total_len, hidden_size, device="cuda")

        # === Reference: Run full sequence through chunked mode ===
        # Force chunk mode by using long sequence or setting mode directly
        model.mode = "chunk"
        with torch.no_grad():
            reference_output = model(full_hidden_states)[0]

        # === Test: Prefill + decode ===
        # Create a simple cache object to hold conv and recurrent states
        class SimpleCache:
            def __init__(self):
                self.conv_states = {0: None}
                self.recurrent_states = {0: None}

        cache = SimpleCache()

        # Prefill phase - force chunk mode
        model.mode = "chunk"
        prefill_input = full_hidden_states[:, :prefill_len, :]
        with torch.no_grad():
            prefill_output = model(
                prefill_input,
                past_key_values=cache,
            )[0]

        # Decode phase - one token at a time (will use fused_recurrent since seq_len=1 <= 64)
        model.mode = "fused_recurrent"  # Ensure recurrent mode for decode
        decode_outputs = []
        for i in range(prefill_len, total_len):
            decode_input = full_hidden_states[:, i : i + 1, :]
            with torch.no_grad():
                decode_output = model(
                    decode_input,
                    past_key_values=cache,
                )[0]
            decode_outputs.append(decode_output)

        # Concatenate prefill + decode outputs
        test_output = torch.cat([prefill_output] + decode_outputs, dim=1)

        # Use looser tolerance for chunked vs recurrent comparison
        # (different processing order leads to numerical differences)
        assert_close(
            test_output,
            reference_output,
            rtol=1e-3,
            atol=1e-3,
            msg=f"KDA chunked vs recurrent mode (prefill={prefill_len}, total={total_len})",
        )

