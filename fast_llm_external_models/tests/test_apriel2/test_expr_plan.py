"""Tests for the expression-based plan system."""

import json
import pytest
import torch

from fast_llm_external_models.apriel2.conversion import (
    Concat,
    EvalKwargs,
    Expr,
    ExprAdapter,
    ExprPlan,
    Init,
    Ref,
    Reshape,
    Slice,
    StreamingExecutor,
    W,
    compose,
    execute,
    fuse,
    full_slice,
    make_slice,
    plan_attention_to_gated_delta_net,
    plan_llava_to_apriel2,
    plan_mil_attention_to_mamba,
    plan_surgery,
    slice_spec,
    substitute,
)


def make_eval_kwargs(
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> EvalKwargs:
    """Create EvalKwargs for testing."""
    return EvalKwargs(
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


class TestExpressionTypes:
    """Test individual expression types."""

    def test_ref_find_refs(self):
        """Ref finds its own key."""
        expr = Ref(key=W("model.weight"))
        assert expr.find_refs() == {W("model.weight")}

    def test_ref_evaluate(self):
        """Ref evaluates to source tensor."""
        expr = Ref(key=W("a"))
        sources = {W("a"): torch.tensor([1.0, 2.0, 3.0])}
        result = expr.evaluate(sources, **make_eval_kwargs())
        assert torch.allclose(result, sources[W("a")])

    def test_ref_missing_key(self):
        """Ref raises KeyError for missing source."""
        expr = Ref(key=W("missing"))
        with pytest.raises(KeyError):
            expr.evaluate({}, **make_eval_kwargs())

    def test_slice_find_refs(self):
        """Slice finds refs from inner expression."""
        expr = Slice(expr=Ref(key=W("a")), slices=((0, 5, None), (None, None, None)))
        assert expr.find_refs() == {W("a")}

    def test_slice_evaluate(self):
        """Slice extracts portion of tensor."""
        expr = Slice(expr=Ref(key=W("a")), slices=((0, 2, None), (1, 3, None)))
        sources = {W("a"): torch.arange(12).reshape(3, 4).float()}
        result = expr.evaluate(sources, **make_eval_kwargs())
        assert result.shape == (2, 2)
        assert torch.allclose(result, torch.tensor([[1, 2], [5, 6]], device=result.device).float())

    def test_concat_find_refs(self):
        """Concat finds refs from all children."""
        expr = Concat(exprs=(Ref(key=W("a")), Ref(key=W("b")), Ref(key=W("c"))), dim=0)
        assert expr.find_refs() == {W("a"), W("b"), W("c")}

    def test_concat_evaluate(self):
        """Concat joins tensors along dimension."""
        expr = Concat(exprs=(Ref(key=W("a")), Ref(key=W("b"))), dim=0)
        sources = {
            W("a"): torch.ones(2, 3),
            W("b"): torch.zeros(3, 3),
        }
        kwargs = make_eval_kwargs()
        result = expr.evaluate(sources, **kwargs)
        assert result.shape == (5, 3)
        # Use result.device for comparisons since Ref preserves source device
        assert torch.allclose(result[:2], torch.ones(2, 3, device=result.device))
        assert torch.allclose(result[2:], torch.zeros(3, 3, device=result.device))

    def test_init_find_refs(self):
        """Init has no refs."""
        expr = Init(shape=(10, 20), init_type="kaiming")
        assert expr.find_refs() == set()

    def test_init_zeros(self):
        """Init zeros creates zero tensor."""
        kwargs = make_eval_kwargs()
        expr = Init(shape=(5, 10), init_type="zeros")
        result = expr.evaluate({}, **kwargs)
        assert result.shape == (5, 10)
        assert torch.allclose(result, torch.zeros(5, 10, device=kwargs["device"], dtype=kwargs["dtype"]))

    def test_init_ones(self):
        """Init ones creates ones tensor."""
        kwargs = make_eval_kwargs()
        expr = Init(shape=(5,), init_type="ones")
        result = expr.evaluate({}, **kwargs)
        assert result.shape == (5,)
        assert torch.allclose(result, torch.ones(5, device=kwargs["device"], dtype=kwargs["dtype"]))

    def test_init_kaiming(self):
        """Init kaiming creates reasonable values."""
        expr = Init(shape=(100, 50), init_type="kaiming")
        result = expr.evaluate({}, **make_eval_kwargs())
        assert result.shape == (100, 50)
        # Kaiming should have reasonable variance
        assert 0.01 < result.std().item() < 1.0

    def test_init_deterministic(self):
        """Init is deterministic given same generator seed."""
        expr = Init(shape=(10, 10), init_type="kaiming")
        result1 = expr.evaluate({}, **make_eval_kwargs(seed=123))
        result2 = expr.evaluate({}, **make_eval_kwargs(seed=123))
        assert torch.allclose(result1, result2)

    def test_init_different_seeds_different_values(self):
        """Different generator seeds give different random values."""
        expr = Init(shape=(10, 10), init_type="kaiming")
        result1 = expr.evaluate({}, **make_eval_kwargs(seed=123))
        result2 = expr.evaluate({}, **make_eval_kwargs(seed=456))
        assert not torch.allclose(result1, result2)

    def test_reshape_find_refs(self):
        """Reshape finds refs from inner expression."""
        expr = Reshape(expr=Ref(key=W("a")), shape=(4, 5))
        assert expr.find_refs() == {W("a")}

    def test_reshape_evaluate(self):
        """Reshape changes tensor shape."""
        expr = Reshape(expr=Ref(key=W("a")), shape=(4, 5))
        sources = {W("a"): torch.arange(20).float()}
        result = expr.evaluate(sources, **make_eval_kwargs())
        assert result.shape == (4, 5)


class TestSliceHelpers:
    """Test slice helper functions."""

    def test_slice_spec(self):
        """slice_spec creates tuple."""
        assert slice_spec(0, 10, 2) == (0, 10, 2)
        assert slice_spec(5, None) == (5, None, None)

    def test_full_slice(self):
        """full_slice creates (None, None, None)."""
        assert full_slice() == (None, None, None)

    def test_make_slice(self):
        """make_slice creates Slice expression."""
        expr = make_slice(Ref(key=W("a")), [slice_spec(0, 5), full_slice()])
        assert isinstance(expr, Slice)
        assert expr.slices == ((0, 5, None), (None, None, None))


class TestSubstitute:
    """Test expression substitution."""

    def test_substitute_ref(self):
        """Substitute replaces Ref with binding."""
        expr = Ref(key=W("x"))
        bindings = {W("x"): Ref(key=W("y"))}
        result = substitute(expr, bindings)
        assert isinstance(result, Ref)
        assert result.key == W("y")

    def test_substitute_ref_passthrough(self):
        """Substitute keeps Ref if no binding."""
        expr = Ref(key=W("x"))
        bindings = {}
        result = substitute(expr, bindings)
        assert result == expr

    def test_substitute_slice(self):
        """Substitute recurses into Slice."""
        expr = Slice(expr=Ref(key=W("x")), slices=((0, 5, None),))
        bindings = {W("x"): Ref(key=W("y"))}
        result = substitute(expr, bindings)
        assert isinstance(result, Slice)
        assert isinstance(result.expr, Ref)
        assert result.expr.key == W("y")

    def test_substitute_concat(self):
        """Substitute recurses into Concat children."""
        expr = Concat(exprs=(Ref(key=W("a")), Ref(key=W("b"))), dim=0)
        bindings = {W("a"): Ref(key=W("x")), W("b"): Ref(key=W("y"))}
        result = substitute(expr, bindings)
        assert isinstance(result, Concat)
        assert result.exprs[0].key == W("x")
        assert result.exprs[1].key == W("y")

    def test_substitute_init_unchanged(self):
        """Substitute leaves Init unchanged."""
        expr = Init(shape=(10,), init_type="zeros")
        result = substitute(expr, {W("x"): Ref(key=W("y"))})
        assert result == expr

    def test_substitute_complex(self):
        """Substitute handles complex nested expressions."""
        # Concat of Slice(Ref) and Init
        expr = Concat(exprs=(
            Slice(expr=Ref(key=W("a")), slices=((0, 5, None),)),
            Init(shape=(5,), init_type="zeros"),
        ), dim=0)
        bindings = {W("a"): Ref(key=W("source"))}
        result = substitute(expr, bindings)

        assert isinstance(result, Concat)
        assert isinstance(result.exprs[0], Slice)
        assert result.exprs[0].expr.key == W("source")
        assert isinstance(result.exprs[1], Init)


class TestFuse:
    """Test expression fusion/optimization."""

    def test_fuse_flatten_concat(self):
        """Fuse flattens nested Concat with same dim."""
        inner = Concat(exprs=(Ref(key=W("a")), Ref(key=W("b"))), dim=0)
        outer = Concat(exprs=(inner, Ref(key=W("c")),), dim=0)
        result = fuse(outer)

        assert isinstance(result, Concat)
        assert len(result.exprs) == 3
        assert result.exprs[0].key == W("a")
        assert result.exprs[1].key == W("b")
        assert result.exprs[2].key == W("c")

    def test_fuse_no_flatten_different_dim(self):
        """Fuse doesn't flatten Concat with different dim."""
        inner = Concat(exprs=(Ref(key=W("a")), Ref(key=W("b"))), dim=1)
        outer = Concat(exprs=(inner, Ref(key=W("c")),), dim=0)
        result = fuse(outer)

        assert isinstance(result, Concat)
        assert len(result.exprs) == 2
        assert isinstance(result.exprs[0], Concat)

    def test_fuse_reshape_reshape(self):
        """Fuse collapses nested Reshape."""
        expr = Reshape(expr=Reshape(expr=Ref(key=W("a")), shape=(4, 5)), shape=(2, 10))
        result = fuse(expr)

        assert isinstance(result, Reshape)
        assert result.shape == (2, 10)
        assert isinstance(result.expr, Ref)


class TestSerialization:
    """Test expression and plan serialization."""

    def test_ref_roundtrip(self):
        """Ref serializes and deserializes."""
        expr = Ref(key=W("model.weight"))
        d = expr.model_dump()
        restored = ExprAdapter.validate_python(d)
        assert isinstance(restored, Ref)
        assert restored.key == expr.key

    def test_slice_roundtrip(self):
        """Slice serializes and deserializes."""
        expr = Slice(expr=Ref(key=W("a")), slices=((0, 5, None), (None, None, 2)))
        d = expr.model_dump()
        restored = ExprAdapter.validate_python(d)
        assert isinstance(restored, Slice)
        assert restored.slices == expr.slices

    def test_concat_roundtrip(self):
        """Concat serializes and deserializes."""
        expr = Concat(exprs=(Ref(key=W("a")), Init(shape=(5,), init_type="zeros")), dim=1)
        d = expr.model_dump()
        restored = ExprAdapter.validate_python(d)
        assert isinstance(restored, Concat)
        assert len(restored.exprs) == 2
        assert restored.dim == 1

    def test_init_roundtrip(self):
        """Init serializes and deserializes."""
        expr = Init(shape=(10, 20), init_type="kaiming")
        d = expr.model_dump()
        restored = ExprAdapter.validate_python(d)
        assert isinstance(restored, Init)
        assert restored.shape == expr.shape
        assert restored.init_type == expr.init_type

    def test_reshape_roundtrip(self):
        """Reshape serializes and deserializes."""
        expr = Reshape(expr=Ref(key=W("a")), shape=(4, 5))
        d = expr.model_dump()
        restored = ExprAdapter.validate_python(d)
        assert isinstance(restored, Reshape)
        assert restored.shape == expr.shape

    def test_plan_json_roundtrip(self):
        """Plan serializes to JSON and back."""
        plan = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                W("out.x"): Ref(key=W("in.x")),
                W("out.y"): Concat(exprs=(Ref(key=W("in.a")), Init(shape=(5,), init_type="zeros")), dim=0),
            },
        )

        d = plan.model_dump()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        restored = ExprPlan.model_validate(d2)

        assert len(restored) == 2
        assert restored.source_format == "a"
        assert restored.target_format == "b"
        assert W("out.x") in restored
        assert W("out.y") in restored


class TestExprPlan:
    """Test ExprPlan class."""

    def test_plan_define_and_access(self):
        """Plan stores and retrieves expressions."""
        plan = ExprPlan(mappings={
            W("target"): Ref(key=W("source")),
        })
        assert W("target") in plan
        assert isinstance(plan[W("target")], Ref)

    def test_plan_source_keys(self):
        """Plan identifies all source references."""
        plan = ExprPlan(mappings={
            W("a"): Ref(key=W("x")),
            W("b"): Concat(exprs=(Ref(key=W("y")), Ref(key=W("z"))), dim=0),
            W("c"): Init(shape=(10,), init_type="zeros"),
        })

        assert plan.source_keys() == {W("x"), W("y"), W("z")}

    def test_plan_target_keys(self):
        """Plan identifies all target keys."""
        plan = ExprPlan(mappings={
            W("a"): Ref(key=W("x")),
            W("b"): Ref(key=W("y")),
        })

        assert plan.target_keys() == {W("a"), W("b")}

    def test_plan_summary(self):
        """Plan summary provides useful info."""
        plan = ExprPlan(
            source_format="llava",
            target_format="apriel2",
            mappings={
                W("a"): Ref(key=W("x")),
                W("b"): Concat(exprs=(Ref(key=W("y")), Ref(key=W("z"))), dim=0),
                W("c"): Init(shape=(10,), init_type="zeros"),
            },
        )

        summary = plan.summary()
        assert summary["source_format"] == "llava"
        assert summary["target_format"] == "apriel2"
        assert summary["num_targets"] == 3
        assert summary["num_source_refs"] == 3

    def test_plan_fuse(self):
        """Plan fuse applies optimizations."""
        inner = Concat(exprs=(Ref(key=W("a")), Ref(key=W("b"))), dim=0)
        plan = ExprPlan(mappings={
            W("out"): Concat(exprs=(inner, Ref(key=W("c")),), dim=0),
        })

        fused = plan.fuse()
        assert isinstance(fused[W("out")], Concat)
        assert len(fused[W("out")].exprs) == 3


class TestComposition:
    """Test plan composition."""

    def test_compose_simple_refs(self):
        """Compose simple Ref chains."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                W("intermediate"): Ref(key=W("original")),
            },
        )

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                W("final"): Ref(key=W("intermediate")),
            },
        )

        composed = plan1 | plan2

        assert composed.source_format == "a"
        assert composed.target_format == "c"
        assert W("final") in composed
        assert isinstance(composed[W("final")], Ref)
        assert composed[W("final")].key == W("original")

    def test_compose_with_concat(self):
        """Compose through Concat expressions."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                W("x"): Ref(key=W("src_x")),
                W("y"): Ref(key=W("src_y")),
            },
        )

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                W("combined"): Concat(exprs=(Ref(key=W("x")), Ref(key=W("y"))), dim=0),
            },
        )

        composed = plan1 | plan2

        assert W("combined") in composed
        result = composed[W("combined")]
        assert isinstance(result, Concat)
        assert result.exprs[0].key == W("src_x")
        assert result.exprs[1].key == W("src_y")

    def test_compose_with_slice(self):
        """Compose through Slice expressions."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                W("full"): Ref(key=W("source")),
            },
        )

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                W("partial"): Slice(expr=Ref(key=W("full")), slices=((0, 5, None),)),
            },
        )

        composed = plan1 | plan2

        result = composed[W("partial")]
        assert isinstance(result, Slice)
        assert isinstance(result.expr, Ref)
        assert result.expr.key == W("source")

    def test_compose_preserves_init(self):
        """Compose preserves Init expressions."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                W("x"): Ref(key=W("src")),
            },
        )

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                W("combined"): Concat(exprs=(Ref(key=W("x")), Init(shape=(5,), init_type="zeros")), dim=0),
            },
        )

        composed = plan1 | plan2

        result = composed[W("combined")]
        assert isinstance(result.exprs[0], Ref)
        assert result.exprs[0].key == W("src")
        assert isinstance(result.exprs[1], Init)

    def test_compose_passthrough(self):
        """Compose keeps refs that plan1 doesn't produce."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                W("x"): Ref(key=W("src_x")),
            },
        )
        # plan1 doesn't define "passthrough"

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                W("out"): Concat(exprs=(Ref(key=W("x")), Ref(key=W("passthrough"))), dim=0),
            },
        )

        composed = plan1 | plan2

        result = composed[W("out")]
        assert result.exprs[0].key == W("src_x")  # Substituted
        assert result.exprs[1].key == W("passthrough")  # Kept as-is


class TestStreamingExecution:
    """Test streaming execution with ref-counting."""

    def test_execute_simple(self):
        """Execute simple plan."""
        plan = ExprPlan(mappings={
            W("out"): Ref(key=W("in")),
        })

        sources = {W("in"): torch.tensor([1.0, 2.0, 3.0])}
        result = execute(plan, sources, seed=42)

        assert W("out") in result
        assert torch.allclose(result[W("out")], sources[W("in")])

    def test_execute_concat(self):
        """Execute plan with Concat."""
        plan = ExprPlan(mappings={
            W("combined"): Concat(exprs=(Ref(key=W("a")), Ref(key=W("b"))), dim=0),
        })

        sources = {
            W("a"): torch.ones(2, 3),
            W("b"): torch.zeros(3, 3),
        }
        result = execute(plan, sources, seed=42)

        assert result[W("combined")].shape == (5, 3)

    def test_execute_mil_like(self):
        """Execute MIL-like Concat of Slices and Init."""
        # Simulated MIL: in_proj = [z, x, B, C]
        plan = ExprPlan(mappings={
            W("in_proj"): Concat(exprs=(
                Init(shape=(4, 8), init_type="zeros"),  # z
                Slice(expr=Ref(key=W("v")), slices=((0, 2, None), (None, None, None))),  # x
                Slice(expr=Ref(key=W("k")), slices=((0, 2, None), (None, None, None))),  # B
                Slice(expr=Ref(key=W("q")), slices=((0, 4, None), (None, None, None))),  # C
            ), dim=0),
        })

        sources = {
            W("q"): torch.ones(4, 8),
            W("k"): torch.full((2, 8), 2.0),
            W("v"): torch.full((2, 8), 3.0),
        }
        result = execute(plan, sources, seed=42)

        assert result[W("in_proj")].shape == (12, 8)
        assert torch.allclose(result[W("in_proj")][0:4], torch.zeros(4, 8))  # z
        assert torch.allclose(result[W("in_proj")][4:6], torch.full((2, 8), 3.0))  # x <- v
        assert torch.allclose(result[W("in_proj")][6:8], torch.full((2, 8), 2.0))  # B <- k
        assert torch.allclose(result[W("in_proj")][8:12], torch.ones(4, 8))  # C <- q

    def test_streaming_execution(self):
        """Streaming executor processes all targets."""
        plan = ExprPlan(mappings={
            W("out1"): Ref(key=W("shared")),
            W("out2"): Ref(key=W("shared")),
            W("out3"): Ref(key=W("unique")),
        })

        load_calls = []

        def loader(key: W) -> torch.Tensor:
            load_calls.append(key)
            return torch.randn(10)

        executor = StreamingExecutor(plan, loader)
        results = list(executor.execute(seed=42))

        # All outputs produced
        assert len(results) == 3
        # Sources loaded (may be called multiple times with mmap, that's fine)
        assert W("shared") in load_calls
        assert W("unique") in load_calls


class TestPlanBuilders:
    """Test plan builder functions."""

    def test_plan_llava_to_apriel2(self, llava_pixtral_config):
        """Llava to Apriel2 plan is built correctly."""
        plan = plan_llava_to_apriel2(llava_pixtral_config)

        assert plan.source_format == "llava"
        assert plan.target_format == "apriel2"
        assert len(plan) > 0

        # Check key mappings exist
        assert "model.embed_tokens.weight" in plan
        assert isinstance(plan["model.embed_tokens.weight"], Ref)

    def test_plan_llava_is_all_refs(self, llava_pixtral_config):
        """Llava plan is pure renaming (all Refs)."""
        plan = plan_llava_to_apriel2(llava_pixtral_config)

        for target, expr in plan:
            assert isinstance(expr, Ref), f"{target} is {type(expr)}, expected Ref"

    def test_plan_mil_attention_to_mamba(self):
        """MIL plan produces correct expressions."""
        exprs = plan_mil_attention_to_mamba(
            layer_idx=0,
            hidden_size=64,
            d_inner=128,
            d_xb=32,
            dt_rank=4,
            d_state=16,
            d_conv=4,
            repeat_kv_before_conv=True,
            conv_bias=True,
            dt_bias=True,
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=0.0001,
            source_prefix=W("model.decoder.blocks.0.mixer"),
            target_prefix=W("model.decoder.blocks.0.mixer"),
        )

        # Check in_proj is Concat
        in_proj = exprs[W("model.decoder.blocks.0.mixer.in_proj.weight")]
        assert isinstance(in_proj, Concat)
        assert len(in_proj.exprs) == 4

        # First is Init (z)
        assert isinstance(in_proj.exprs[0], Init)
        assert in_proj.exprs[0].shape == (128, 64)

        # Others are Slices of attention weights
        assert isinstance(in_proj.exprs[1], Slice)  # x <- v
        assert isinstance(in_proj.exprs[2], Slice)  # B <- k
        assert isinstance(in_proj.exprs[3], Slice)  # C <- q

        # out_proj is direct Ref
        out_proj = exprs[W("model.decoder.blocks.0.mixer.out_proj.weight")]
        assert isinstance(out_proj, Ref)

    def test_plan_mil_execution(self):
        """MIL plan executes correctly with actual weights."""
        plan = plan_mil_attention_to_mamba(
            layer_idx=0,
            hidden_size=64,
            d_inner=128,
            d_xb=32,
            dt_rank=4,
            d_state=16,
            d_conv=4,
            repeat_kv_before_conv=True,
            conv_bias=True,
            dt_bias=True,
            dt_min=0.001,
            dt_max=0.1,
            dt_init_floor=0.0001,
            source_prefix=W("attn"),
            target_prefix=W("mamba"),
        )

        # Create attention weights
        sources = {
            W("attn.q_proj.weight"): torch.full((128, 64), 1.0),
            W("attn.k_proj.weight"): torch.full((32, 64), 2.0),
            W("attn.v_proj.weight"): torch.full((32, 64), 3.0),
            W("attn.o_proj.weight"): torch.full((64, 128), 4.0),
        }

        result = execute(plan, sources, seed=42)

        # Verify in_proj layout: [z, x, B, C]
        in_proj = result[W("mamba.in_proj.weight")]
        assert in_proj.shape == (128 + 32 + 32 + 128, 64)

        # z (0:128) is random init
        # x (128:160) should be 3.0 (from v)
        assert torch.allclose(in_proj[128:160], torch.full((32, 64), 3.0))
        # B (160:192) should be 2.0 (from k)
        assert torch.allclose(in_proj[160:192], torch.full((32, 64), 2.0))
        # C (192:320) should be 1.0 (from q)
        assert torch.allclose(in_proj[192:320], torch.full((128, 64), 1.0))

        # out_proj should be 4.0
        assert torch.allclose(result[W("mamba.out_proj.weight")], torch.full((64, 128), 4.0))

    def test_plan_attention_to_gated_delta_net(self):
        """DIL plan produces correct per-head-group interleaved structure."""
        # MHA case: num_v_heads == num_k_heads (no GQA), 1 v_head per group
        plan = plan_attention_to_gated_delta_net(
            hidden_size=64,
            num_v_heads=4,
            num_k_heads=4,
            head_k_dim=16,
            head_v_dim=16,
            conv_kernel_size=4,
            source_num_q_heads=4,
            source_num_kv_heads=4,
            source_head_dim=16,
            source_prefix=W("attn"),
            target_prefix=W(""),
        )

        # Calculate expected dimensions
        key_dim = 4 * 16  # 64
        value_dim = 4 * 16  # 64
        conv_dim = 2 * key_dim + value_dim  # 192

        # Check in_proj_qkvz is Concat of 4 head groups
        in_proj_qkvz = plan[W("in_proj_qkvz.weight")]
        assert isinstance(in_proj_qkvz, Concat)
        assert len(in_proj_qkvz.exprs) == 4  # 4 head groups

        # Each group should be Concat of [Q_head, K_head, V_head, Z_head]
        for g, group in enumerate(in_proj_qkvz.exprs):
            assert isinstance(group, Concat), f"Group {g} should be Concat"
            assert len(group.exprs) == 4, f"Group {g} should have 4 parts"

            # Q: Slice from q_proj for head g
            assert isinstance(group.exprs[0], Slice)
            # K: Slice from k_proj for head g
            assert isinstance(group.exprs[1], Slice)
            # V: Slice from v_proj (single head in MHA)
            assert isinstance(group.exprs[2], Slice)
            # Z: Init zeros
            assert isinstance(group.exprs[3], Init)
            assert group.exprs[3].init_type == "zeros"

        # Check in_proj_ba: zeros, shape (2*num_v_heads, hidden_size)
        in_proj_ba = plan[W("in_proj_ba.weight")]
        assert isinstance(in_proj_ba, Init)
        assert in_proj_ba.shape == (2 * 4, 64)  # (8, 64)
        assert in_proj_ba.init_type == "zeros"

        # Check out_proj: direct Ref to o_proj
        out_proj = plan[W("out_proj.weight")]
        assert isinstance(out_proj, Ref)
        assert "o_proj" in out_proj.key

        # Check conv1d: scaled identity kernel (0.5 for SiLU linearity)
        conv1d = plan[W("convolution.weight")]
        assert isinstance(conv1d, Init)
        assert conv1d.shape == (conv_dim, 1, 4)
        assert conv1d.init_type == "scaled_identity_conv"

        # Check A_log: slow decay
        a_log = plan[W("A_log")]
        assert isinstance(a_log, Init)
        assert a_log.shape == (4,)  # num_v_heads
        assert a_log.init_type == "slow_decay"

        # Check dt_bias: zeros
        dt_bias = plan[W("dt_bias")]
        assert isinstance(dt_bias, Init)
        assert dt_bias.shape == (4,)  # num_v_heads
        assert dt_bias.init_type == "zeros"

        # Check norm.weight: ones
        norm_weight = plan[W("norm.weight")]
        assert isinstance(norm_weight, Init)
        assert norm_weight.shape == (16,)  # head_v_dim
        assert norm_weight.init_type == "ones"

    def test_plan_attention_to_gated_delta_net_gqa(self):
        """DIL plan handles GQA with tiling (not padding)."""
        # GQA case: 4 v_heads, 2 k_heads → 2 v_heads per group
        # Source has 4 Q heads, 2 KV heads
        plan = plan_attention_to_gated_delta_net(
            hidden_size=64,
            num_v_heads=4,
            num_k_heads=2,
            head_k_dim=16,
            head_v_dim=16,
            conv_kernel_size=4,
            source_num_q_heads=4,
            source_num_kv_heads=2,
            source_head_dim=16,
            source_prefix=W("attn"),
            target_prefix=W(""),
        )

        # Check in_proj_qkvz is Concat of 2 head groups
        in_proj_qkvz = plan[W("in_proj_qkvz.weight")]
        assert isinstance(in_proj_qkvz, Concat)
        assert len(in_proj_qkvz.exprs) == 2  # 2 k_head groups

        # Each group has 2 v_heads, so V should be Concat of 2 slices
        for g, group in enumerate(in_proj_qkvz.exprs):
            assert isinstance(group, Concat), f"Group {g} should be Concat"
            assert len(group.exprs) == 4  # [Q, K, V_group, Z]

            # V_group should be Concat of 2 v_head slices (tiled from source)
            v_group = group.exprs[2]
            assert isinstance(v_group, Concat), f"V_group {g} should be Concat"
            assert len(v_group.exprs) == 2  # 2 v_heads per group

            # Both should be Slices (tiled from source heads via modulo)
            for v_slice in v_group.exprs:
                assert isinstance(v_slice, Slice)

    def test_plan_dil_execution(self):
        """DIL plan executes correctly with per-head-group interleaving."""
        # MHA case: 4 k_heads, 4 v_heads (1 v_head per group)
        plan = plan_attention_to_gated_delta_net(
            hidden_size=64,
            num_v_heads=4,
            num_k_heads=4,
            head_k_dim=16,
            head_v_dim=16,
            conv_kernel_size=4,
            source_num_q_heads=4,
            source_num_kv_heads=4,
            source_head_dim=16,
            source_prefix=W("attn"),
            target_prefix=W(""),
        )

        key_dim = 64
        value_dim = 64
        head_k_dim = 16
        head_v_dim = 16
        conv_dim = 192

        # Create attention weights with per-head distinctive values
        # Q: each head gets value (head_idx + 1)
        q_weight = torch.zeros(64, 64)
        for h in range(4):
            q_weight[h*16:(h+1)*16, :] = float(h + 1)

        # K: each head gets value (head_idx + 1) * 10
        k_weight = torch.zeros(64, 64)
        for h in range(4):
            k_weight[h*16:(h+1)*16, :] = float((h + 1) * 10)

        # V: each head gets value (head_idx + 1) * 100
        v_weight = torch.zeros(64, 64)
        for h in range(4):
            v_weight[h*16:(h+1)*16, :] = float((h + 1) * 100)

        sources = {
            W("attn.q_proj.weight"): q_weight,
            W("attn.k_proj.weight"): k_weight,
            W("attn.v_proj.weight"): v_weight,
            W("attn.o_proj.weight"): torch.full((64, 64), 4.0),
        }

        result = execute(plan, sources, seed=42)

        # Verify in_proj_qkvz has per-head-group interleaved layout
        in_proj_qkvz = result[W("in_proj_qkvz.weight")]
        # Total: 4 groups * (16 + 16 + 16 + 16) = 256
        assert in_proj_qkvz.shape == (256, 64)

        # Check each group: [Q_h, K_h, V_h, Z_h]
        group_size = 16 + 16 + 16 + 16  # 64 per group
        for g in range(4):
            base = g * group_size
            # Q_h (rows 0-15 in group)
            assert torch.allclose(in_proj_qkvz[base:base+16], torch.full((16, 64), float(g + 1)))
            # K_h (rows 16-31 in group)
            assert torch.allclose(in_proj_qkvz[base+16:base+32], torch.full((16, 64), float((g + 1) * 10)))
            # V_h (rows 32-47 in group)
            assert torch.allclose(in_proj_qkvz[base+32:base+48], torch.full((16, 64), float((g + 1) * 100)))
            # Z_h (rows 48-63 in group) - zeros
            assert torch.allclose(in_proj_qkvz[base+48:base+64], torch.zeros(16, 64))

        # in_proj_ba should be zeros
        in_proj_ba = result[W("in_proj_ba.weight")]
        assert in_proj_ba.shape == (8, 64)
        assert torch.allclose(in_proj_ba, torch.zeros(8, 64))

        # out_proj should be 4.0 (direct copy)
        assert torch.allclose(result[W("out_proj.weight")], torch.full((64, 64), 4.0))

        # conv1d should be scaled identity kernel (0.5 at last position)
        conv1d = result[W("convolution.weight")]
        assert conv1d.shape == (conv_dim, 1, 4)
        expected_conv = torch.zeros(conv_dim, 1, 4)
        expected_conv[:, 0, -1] = 0.5  # Scaled for SiLU linearity
        assert torch.allclose(conv1d, expected_conv)

        # A_log should be log(0.1) ≈ -2.3
        a_log = result[W("A_log")]
        assert a_log.shape == (4,)
        assert torch.allclose(a_log, torch.full((4,), -2.302585), atol=1e-5)

        # dt_bias should be zeros
        dt_bias = result[W("dt_bias")]
        assert dt_bias.shape == (4,)
        assert torch.allclose(dt_bias, torch.zeros(4))

        # norm.weight should be ones
        norm_weight = result[W("norm.weight")]
        assert norm_weight.shape == (16,)
        assert torch.allclose(norm_weight, torch.ones(16))

    def test_plan_dil_execution_gqa(self):
        """DIL plan executes correctly with GQA (V heads tiled via modulo)."""
        # GQA: 4 v_heads, 2 k_heads → 2 v_heads per group
        # Source: 4 Q heads, 2 KV heads
        plan = plan_attention_to_gated_delta_net(
            hidden_size=64,
            num_v_heads=4,
            num_k_heads=2,
            head_k_dim=16,
            head_v_dim=16,
            conv_kernel_size=4,
            source_num_q_heads=4,
            source_num_kv_heads=2,
            source_head_dim=16,
            source_prefix=W("attn"),
            target_prefix=W(""),
        )

        # Create attention weights
        # Q: 4 heads, each with value (head_idx + 1)
        q_weight = torch.zeros(64, 64)
        for h in range(4):
            q_weight[h*16:(h+1)*16, :] = float(h + 1)

        # K: 2 kv_heads, each with value (head_idx + 1) * 10
        k_weight = torch.zeros(32, 64)
        for h in range(2):
            k_weight[h*16:(h+1)*16, :] = float((h + 1) * 10)

        # V: 2 kv_heads, each with value (head_idx + 1) * 100
        v_weight = torch.zeros(32, 64)
        for h in range(2):
            v_weight[h*16:(h+1)*16, :] = float((h + 1) * 100)

        sources = {
            W("attn.q_proj.weight"): q_weight,
            W("attn.k_proj.weight"): k_weight,
            W("attn.v_proj.weight"): v_weight,
            W("attn.o_proj.weight"): torch.full((64, 64), 4.0),
        }

        result = execute(plan, sources, seed=42)

        # Verify in_proj_qkvz with GQA tiling
        in_proj_qkvz = result[W("in_proj_qkvz.weight")]
        # 2 groups * (16 + 16 + 32 + 32) = 2 * 96 = 192
        v_per_group = 2
        group_size = 16 + 16 + v_per_group * 16 + v_per_group * 16  # 96 per group
        assert in_proj_qkvz.shape == (192, 64)

        # Group 0: Q from head 0, K from kv_head 0, V from kv_heads 0,1 (tiled)
        base = 0
        # Q_0 (maps to source Q head 0)
        assert torch.allclose(in_proj_qkvz[base:base+16], torch.full((16, 64), 1.0))
        # K_0 (maps to source K head 0)
        assert torch.allclose(in_proj_qkvz[base+16:base+32], torch.full((16, 64), 10.0))
        # V_group_0: v_heads 0,1 → source v_heads 0,1 (via modulo)
        # v_head 0 → src_v_head 0 (value 100)
        assert torch.allclose(in_proj_qkvz[base+32:base+48], torch.full((16, 64), 100.0))
        # v_head 1 → src_v_head 1 (value 200)
        assert torch.allclose(in_proj_qkvz[base+48:base+64], torch.full((16, 64), 200.0))
        # Z_group_0: zeros
        assert torch.allclose(in_proj_qkvz[base+64:base+96], torch.zeros(32, 64))

        # Group 1: Q from head 1, K from kv_head 1, V from kv_heads 2,3 (tiled to 0,1)
        base = 96
        # Q_1 (maps to source Q head 1)
        assert torch.allclose(in_proj_qkvz[base:base+16], torch.full((16, 64), 2.0))
        # K_1 (maps to source K head 1)
        assert torch.allclose(in_proj_qkvz[base+16:base+32], torch.full((16, 64), 20.0))
        # V_group_1: v_heads 2,3 → source v_heads 0,1 (via modulo, tiled)
        # v_head 2 → src_v_head 0 (value 100)
        assert torch.allclose(in_proj_qkvz[base+32:base+48], torch.full((16, 64), 100.0))
        # v_head 3 → src_v_head 1 (value 200)
        assert torch.allclose(in_proj_qkvz[base+48:base+64], torch.full((16, 64), 200.0))
        # Z_group_1: zeros
        assert torch.allclose(in_proj_qkvz[base+64:base+96], torch.zeros(32, 64))


class TestFullPipeline:
    """Test full conversion + surgery pipeline."""

    def test_compose_llava_to_mamba(self, llava_pixtral_config, apriel2_config_stochastic):
        """Can compose Llava conversion with surgery to stochastic."""
        # Build conversion plan
        conversion_plan = plan_llava_to_apriel2(llava_pixtral_config)

        # Build surgery plan (need intermediate config)
        from fast_llm_external_models.apriel2.conversion.llava import convert_config
        intermediate_config = convert_config(llava_pixtral_config)
        target_config = apriel2_config_stochastic.to_dict()
        surgery_plan = plan_surgery(intermediate_config, target_config)

        # Compose using | operator
        full_plan = conversion_plan | surgery_plan

        assert full_plan.source_format == "llava"
        assert full_plan.target_format == "apriel2"

        # Should have fused through to llava sources
        summary = full_plan.summary()
        assert summary["num_targets"] > 0

    def test_execute_composed_pipeline(self, llava_pixtral_checkpoint):
        """Execute composed conversion pipeline on checkpoint (without surgery).

        Note: Full surgery execution requires matching dimensions between
        test fixtures. This test verifies the conversion portion works.
        """
        import json
        from pathlib import Path
        from safetensors.torch import load_file

        # Load config
        with open(Path(llava_pixtral_checkpoint) / "config.json") as f:
            llava_config = json.load(f)

        # Build conversion plan only (surgery tested separately in test_compose_llava_to_mamba)
        conversion_plan = plan_llava_to_apriel2(llava_config)

        # Load source weights
        source_weights = load_file(str(Path(llava_pixtral_checkpoint) / "model.safetensors"))

        # Execute conversion
        result = execute(conversion_plan, source_weights, seed=42)

        assert len(result) > 0

        # Verify key mappings worked
        assert "model.embed_tokens.weight" in result
        assert any("mixer.q_proj" in k for k in result)


class TestExpressionRepr:
    """Test expression string representations."""

    def test_ref_repr(self):
        """Ref has readable repr."""
        expr = Ref(key=W("model.weight"))
        assert "model.weight" in repr(expr)

    def test_slice_repr(self):
        """Slice has readable repr."""
        expr = Slice(expr=Ref(key=W("a")), slices=((0, 5, None), (None, None, None)))
        r = repr(expr)
        # Repr shows :5 for 0:5 (standard Python slice notation)
        assert ":5" in r
        assert ":" in r

    def test_concat_repr(self):
        """Concat has readable repr."""
        expr = Concat(exprs=(Ref(key=W("a")), Ref(key=W("b"))), dim=0)
        r = repr(expr)
        assert "Concat" in r
        assert "dim=0" in r

    def test_init_repr(self):
        """Init has readable repr."""
        expr = Init(shape=(10, 20), init_type="kaiming")
        r = repr(expr)
        assert "(10, 20)" in r
        assert "kaiming" in r


class TestInitModeSemantics:
    """Test init: transfer vs init: random semantics in surgery."""

    def test_transfer_fails_for_unsupported_conversion(self):
        """init: transfer (default) fails fast when no converter exists."""
        # Source config with mamba
        source_config = {
            "hidden_size": 64,
            "vocab_size": 100,
            "decoder": {
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "mamba",
                        "d_inner": 128,
                        "d_state": 16,
                        "dt_rank": 4,
                        "d_xb": 32,
                        "d_conv": 4,
                        "repeat_kv_before_conv": True,
                        "conv_bias": True,
                        "dt_proj_bias": True,
                        "dt_min": 0.001,
                        "dt_max": 0.1,
                        "dt_init_floor": 1e-4,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

        # Target with gated_delta_net - no mamba->GDN converter exists
        target_config = {
            **source_config,
            "decoder": {
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "gdn",
                        "init": "transfer",  # explicitly request transfer
                        "value_heads": 4,
                        "key_heads": 2,
                        "key_head_dim": 16,
                        "value_head_dim": 16,
                        "conv_kernel_size": 4,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

        with pytest.raises(ValueError, match="No converter available"):
            plan_surgery(source_config, target_config)

    def test_random_succeeds_for_unsupported_conversion(self):
        """init: random allows any target type without converter."""
        # Source config with mamba (no converter to GDN exists)
        source_config = {
            "hidden_size": 64,
            "vocab_size": 100,
            "decoder": {
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "mamba",
                        "d_inner": 128,
                        "d_state": 16,
                        "dt_rank": 4,
                        "d_xb": 32,
                        "d_conv": 4,
                        "repeat_kv_before_conv": True,
                        "conv_bias": True,
                        "dt_proj_bias": True,
                        "dt_min": 0.001,
                        "dt_max": 0.1,
                        "dt_init_floor": 1e-4,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

        # Target with gated_delta_net using random init (requires explicit params)
        target_config = {
            **source_config,
            "decoder": {
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "gdn",
                        "init": "random",  # random init - no converter needed
                        "value_heads": 4,
                        "key_heads": 2,
                        "key_head_dim": 16,
                        "value_head_dim": 16,
                        "conv_kernel_size": 4,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

        # Should succeed - random init doesn't need a converter
        plan = plan_surgery(source_config, target_config)
        assert len(plan) > 0

    def test_transfer_default_for_supported_conversion(self):
        """Default (no init key) uses transfer for supported conversions."""
        source_config = {
            "hidden_size": 64,
            "vocab_size": 100,
            "decoder": {
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

        # Target with attention (same type) - no init key
        target_config = {
            **source_config,
            "decoder": {
                "type": "fixed",
                "num_blocks": 1,
                "block": {
                    "mixer": {
                        "type": "attention",
                        "heads": 4,
                        "head_groups": 2,
                        "head_size": 16,
                        # No init key - defaults to transfer
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

        plan = plan_surgery(source_config, target_config)

        # Verify it uses Refs (transfer), not Init (random)
        for target, expr in plan:
            if "self_attn" in target:
                assert isinstance(expr, Ref), f"Expected Ref for {target}, got {type(expr)}"


class TestEndToEndConversion:
    """End-to-end conversion tests that validate against actual Apriel2 model loading.

    The ultimate validation: if converted weights load into an Apriel2 model
    with strict=True, then all keys and shapes are correct.
    """

    def test_comprehensive_conversion_all_mixer_types(self, llava_pixtral_checkpoint, tmp_path):
        """Full pipeline: LLaVA → Apriel2 with surgery exercising ALL conversion paths.

        This test creates a comprehensive surgery config with:
        - Layer 0: Attention → Attention (passthrough)
        - Layer 1: Attention → Mamba (MIL conversion)
        - Layer 2: Attention → GatedDeltaNet (DIL conversion)
        - Layer 3: Attention → Stochastic(Attention + Mamba)
        - Layer 4: Attention → Stochastic(SWA + GDN)

        The validation is simple: if load_state_dict(strict=True) works,
        the conversion produced correct keys and shapes.
        """
        import json
        from pathlib import Path

        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
        from fast_llm_external_models.apriel2.convert import build_plan, convert
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration

        # Load LLaVA config
        with open(llava_pixtral_checkpoint / "config.json") as f:
            llava_config = json.load(f)

        # Get source dimensions for surgery config
        text_config = llava_config["text_config"]
        hidden_size = text_config["hidden_size"]  # 256
        num_heads = text_config["num_attention_heads"]  # 8
        num_kv_heads = text_config["num_key_value_heads"]  # 4
        head_size = hidden_size // num_heads  # 32

        # Create comprehensive surgery config exercising ALL conversion paths
        surgery_config = {
            "hidden_size": hidden_size,
            "vocab_size": text_config["vocab_size"],
            "bos_token_id": text_config.get("bos_token_id", 1),
            "eos_token_id": text_config.get("eos_token_id", 2),
            "tie_word_embeddings": text_config.get("tie_word_embeddings", False),
            "image_token_index": llava_config["image_token_index"],
            "decoder": {
                "type": "pattern",
                "num_blocks": 5,
                "pattern": [
                    "attn",        # 0: attention → attention (passthrough)
                    "mamba",       # 1: attention → mamba (MIL)
                    "gdn",         # 2: attention → gated_delta_net (DIL)
                    "stoch_am",    # 3: attention → stochastic(attention + mamba)
                    "stoch_sg",    # 4: attention → stochastic(swa + gdn)
                ],
                "blocks": {
                    # Pure attention (passthrough from source)
                    "attn": {
                        "mixer": {
                            "type": "attention",
                            "heads": num_heads,
                            "head_groups": num_kv_heads,
                            "head_size": head_size,
                            "rotary": {"type": "mistral_1d", "theta": text_config["rope_theta"]},
                        },
                        "mlp": {"type": "mlp", "intermediate_size": text_config["intermediate_size"]},
                        "normalization": {"type": "rms_norm", "epsilon": text_config["rms_norm_eps"]},
                    },
                    # Pure Mamba (MIL conversion from attention)
                    # MIL requires Mamba dims to match attention dims:
                    # - d_inner = num_heads * head_size (for Q -> C mapping)
                    # - d_xb = num_kv_heads * head_size (for K -> B, V -> x mapping)
                    "mamba": {
                        "mixer": {
                            "type": "mamba",
                            "d_inner": num_heads * head_size,  # 256, matches Q
                            "d_state": 16,
                            "dt_rank": hidden_size // 16,
                            "d_xb": num_kv_heads * head_size,  # 128, matches K/V
                            "d_conv": 4,
                            "repeat_kv_before_conv": True,
                            "conv_bias": True,
                            "dt_proj_bias": True,
                            "dt_min": 0.001,
                            "dt_max": 0.1,
                            "dt_init_floor": 1e-4,
                        },
                        "mlp": {"type": "mlp", "intermediate_size": text_config["intermediate_size"]},
                        "normalization": {"type": "rms_norm", "epsilon": text_config["rms_norm_eps"]},
                    },
                    # Pure GatedDeltaNet (DIL conversion from attention)
                    "gdn": {
                        "mixer": {
                            "type": "gdn",
                            "value_heads": num_heads,
                            "key_heads": num_kv_heads,
                            "key_head_dim": head_size,
                            "value_head_dim": head_size,
                            "conv_kernel_size": 4,
                        },
                        "mlp": {"type": "mlp", "intermediate_size": text_config["intermediate_size"]},
                        "normalization": {"type": "rms_norm", "epsilon": text_config["rms_norm_eps"]},
                    },
                    # Stochastic: attention + mamba
                    "stoch_am": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {
                                "attention": {
                                    "type": "attention",
                                    "heads": num_heads,
                                    "head_groups": num_kv_heads,
                                    "head_size": head_size,
                                    "rotary": {"type": "mistral_1d", "theta": text_config["rope_theta"]},
                                },
                                "mamba": {
                                    "type": "mamba",
                                    "d_inner": num_heads * head_size,  # matches Q
                                    "d_state": 16,
                                    "dt_rank": hidden_size // 16,
                                    "d_xb": num_kv_heads * head_size,  # matches K/V
                                    "d_conv": 4,
                                    "repeat_kv_before_conv": True,
                                    "conv_bias": True,
                                    "dt_proj_bias": True,
                                    "dt_min": 0.001,
                                    "dt_max": 0.1,
                                    "dt_init_floor": 1e-4,
                                },
                            },
                        },
                        "mlp": {"type": "mlp", "intermediate_size": text_config["intermediate_size"]},
                        "normalization": {"type": "rms_norm", "epsilon": text_config["rms_norm_eps"]},
                    },
                    # Stochastic: sliding window attention + gated delta net
                    "stoch_sg": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "swa",
                            "mixers": {
                                "swa": {
                                    "type": "attention",
                                    "heads": num_heads,
                                    "head_groups": num_kv_heads,
                                    "head_size": head_size,
                                    "sliding_window": 512,
                                    "rotary": {"type": "mistral_1d", "theta": text_config["rope_theta"]},
                                },
                                "gdn": {
                                    "type": "gdn",
                                    "value_heads": num_heads,
                                    "key_heads": num_kv_heads,
                                    "key_head_dim": head_size,
                                    "value_head_dim": head_size,
                                    "conv_kernel_size": 4,
                                },
                            },
                        },
                        "mlp": {"type": "mlp", "intermediate_size": text_config["intermediate_size"]},
                        "normalization": {"type": "rms_norm", "epsilon": text_config["rms_norm_eps"]},
                    },
                },
            },
            # Vision encoder config (passthrough)
            "vision_encoder": {
                "hidden_size": llava_config["vision_config"]["hidden_size"],
                "embeddings": {
                    "patch_height": llava_config["vision_config"]["patch_size"],
                    "patch_width": llava_config["vision_config"]["patch_size"],
                    "input_channels": llava_config["vision_config"]["num_channels"],
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
                "encoder": {
                    "type": "fixed",
                    "num_blocks": llava_config["vision_config"]["num_hidden_layers"],
                    "block": {
                        "mixer": {
                            "type": "attention",
                            "heads": llava_config["vision_config"]["num_attention_heads"],
                            "head_groups": llava_config["vision_config"]["num_attention_heads"],
                            "head_size": llava_config["vision_config"]["hidden_size"] // llava_config["vision_config"]["num_attention_heads"],
                            "add_linear_biases": False,
                            "causal": False,
                            "rotary": {
                                "type": "pixtral_2d",
                                "theta": llava_config["vision_config"]["rope_theta"],
                                "max_image_size": llava_config["vision_config"]["image_size"],
                                "patch_size": llava_config["vision_config"]["patch_size"],
                            },
                        },
                        "mlp": {
                            "type": "mlp",
                            "intermediate_size": llava_config["vision_config"]["intermediate_size"],
                            "activation": llava_config["vision_config"]["hidden_act"],
                            "gated": True,
                            "add_linear_biases": False,
                        },
                        "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                    },
                },
                "adapter": {
                    "type": "mlp",
                    "intermediate_size": hidden_size,
                    "activation": llava_config["projector_hidden_act"],
                    "add_linear_biases": True,
                },
            },
        }

        # Run conversion
        output_dir = tmp_path / "converted"
        output_dir.mkdir()

        safetensor_files = sorted(llava_pixtral_checkpoint.glob("*.safetensors"))
        final_config = convert(
            llava_config,
            safetensor_files,
            output_dir,
            surgery_configs=[surgery_config],
        )

        # Save config for model loading
        with open(output_dir / "config.json", "w") as f:
            json.dump(final_config, f)

        # THE ULTIMATE VALIDATION: Load into Apriel2 model
        # If this works with strict=True, all keys and shapes are correct
        from safetensors.torch import load_file

        # Load converted weights
        converted_files = sorted(output_dir.glob("*.safetensors"))
        converted_weights = {}
        for f in converted_files:
            converted_weights.update(load_file(f))

        # Create Apriel2 model with the surgery config
        apriel2_config = Apriel2Config(**final_config)
        model = Apriel2ForConditionalGeneration(apriel2_config)

        # This is the key validation - strict=True means all keys must match
        missing_keys, unexpected_keys = model.load_state_dict(converted_weights, strict=False)

        # Assert no missing or unexpected keys
        assert not missing_keys, f"Missing keys in converted weights: {missing_keys}"
        assert not unexpected_keys, f"Unexpected keys in converted weights: {unexpected_keys}"

        # Bonus: verify we can run a forward pass
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, surgery_config["vocab_size"], (1, 10))
            outputs = model(input_ids, use_cache=False)
            assert outputs.logits.shape == (1, 10, surgery_config["vocab_size"])

    def test_conversion_plan_targets_match_model_state_dict(self, llava_pixtral_config):
        """Verify that plan target keys exactly match model state_dict keys.

        This test validates the plan WITHOUT executing it, by comparing
        plan target keys against what the model expects.
        """
        import json

        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
        from fast_llm_external_models.apriel2.convert import build_plan
        from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration

        # Build plan for simple LLaVA -> Apriel2 conversion (no surgery)
        plan, final_config = build_plan(llava_pixtral_config)

        # Create model to get expected keys
        apriel2_config = Apriel2Config(**final_config)
        model = Apriel2ForConditionalGeneration(apriel2_config)
        expected_keys = set(model.state_dict().keys())

        # Get plan target keys
        plan_target_keys = set(str(k) for k in plan.target_keys())

        # Compare
        missing_from_plan = expected_keys - plan_target_keys
        extra_in_plan = plan_target_keys - expected_keys

        assert not missing_from_plan, f"Plan missing keys that model expects: {sorted(missing_from_plan)}"
        assert not extra_in_plan, f"Plan has extra keys model doesn't expect: {sorted(extra_in_plan)}"
