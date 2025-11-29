"""Tests for the expression-based plan system."""

import json
import pytest
import torch

from fast_llm_external_models.apriel2.expr_plan import (
    Concat,
    Expr,
    ExprAdapter,
    ExprPlan,
    Init,
    Ref,
    Reshape,
    Slice,
    StreamingExecutor,
    compose,
    execute,
    fuse,
    full_slice,
    make_slice,
    plan_llava_to_apriel2,
    plan_mil_attention_to_mamba,
    plan_surgery,
    slice_spec,
    substitute,
)


class TestExpressionTypes:
    """Test individual expression types."""

    def test_ref_find_refs(self):
        """Ref finds its own key."""
        expr = Ref(key="model.weight")
        assert expr.find_refs() == {"model.weight"}

    def test_ref_evaluate(self):
        """Ref evaluates to source tensor."""
        expr = Ref(key="a")
        sources = {"a": torch.tensor([1.0, 2.0, 3.0])}
        result = expr.evaluate(sources)
        assert torch.allclose(result, sources["a"])

    def test_ref_missing_key(self):
        """Ref raises KeyError for missing source."""
        expr = Ref(key="missing")
        with pytest.raises(KeyError):
            expr.evaluate({})

    def test_slice_find_refs(self):
        """Slice finds refs from inner expression."""
        expr = Slice(expr=Ref(key="a"), slices=((0, 5, None), (None, None, None)))
        assert expr.find_refs() == {"a"}

    def test_slice_evaluate(self):
        """Slice extracts portion of tensor."""
        expr = Slice(expr=Ref(key="a"), slices=((0, 2, None), (1, 3, None)))
        sources = {"a": torch.arange(12).reshape(3, 4).float()}
        result = expr.evaluate(sources)
        assert result.shape == (2, 2)
        assert torch.allclose(result, torch.tensor([[1, 2], [5, 6]]).float())

    def test_concat_find_refs(self):
        """Concat finds refs from all children."""
        expr = Concat(exprs=(Ref(key="a"), Ref(key="b"), Ref(key="c")), dim=0)
        assert expr.find_refs() == {"a", "b", "c"}

    def test_concat_evaluate(self):
        """Concat joins tensors along dimension."""
        expr = Concat(exprs=(Ref(key="a"), Ref(key="b")), dim=0)
        sources = {
            "a": torch.ones(2, 3),
            "b": torch.zeros(3, 3),
        }
        result = expr.evaluate(sources)
        assert result.shape == (5, 3)
        assert torch.allclose(result[:2], torch.ones(2, 3))
        assert torch.allclose(result[2:], torch.zeros(3, 3))

    def test_init_find_refs(self):
        """Init has no refs."""
        expr = Init(shape=(10, 20), init_type="kaiming")
        assert expr.find_refs() == set()

    def test_init_zeros(self):
        """Init zeros creates zero tensor."""
        expr = Init(shape=(5, 10), init_type="zeros")
        result = expr.evaluate({})
        assert result.shape == (5, 10)
        assert torch.allclose(result, torch.zeros(5, 10))

    def test_init_ones(self):
        """Init ones creates ones tensor."""
        expr = Init(shape=(5,), init_type="ones")
        result = expr.evaluate({})
        assert result.shape == (5,)
        assert torch.allclose(result, torch.ones(5))

    def test_init_kaiming(self):
        """Init kaiming creates reasonable values."""
        expr = Init(shape=(100, 50), init_type="kaiming")
        result = expr.evaluate({})
        assert result.shape == (100, 50)
        # Kaiming should have reasonable variance
        assert 0.01 < result.std().item() < 1.0

    def test_init_deterministic(self):
        """Init is deterministic given target key."""
        expr = Init(shape=(10, 10), init_type="kaiming")
        result1 = expr.evaluate({}, target_key="model.layer.weight")
        result2 = expr.evaluate({}, target_key="model.layer.weight")
        assert torch.allclose(result1, result2)

    def test_init_different_keys_different_values(self):
        """Different target keys give different random values."""
        expr = Init(shape=(10, 10), init_type="kaiming")
        result1 = expr.evaluate({}, target_key="model.layer1.weight")
        result2 = expr.evaluate({}, target_key="model.layer2.weight")
        assert not torch.allclose(result1, result2)

    def test_reshape_find_refs(self):
        """Reshape finds refs from inner expression."""
        expr = Reshape(expr=Ref(key="a"), shape=(4, 5))
        assert expr.find_refs() == {"a"}

    def test_reshape_evaluate(self):
        """Reshape changes tensor shape."""
        expr = Reshape(expr=Ref(key="a"), shape=(4, 5))
        sources = {"a": torch.arange(20).float()}
        result = expr.evaluate(sources)
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
        expr = make_slice(Ref(key="a"), [slice_spec(0, 5), full_slice()])
        assert isinstance(expr, Slice)
        assert expr.slices == ((0, 5, None), (None, None, None))


class TestSubstitute:
    """Test expression substitution."""

    def test_substitute_ref(self):
        """Substitute replaces Ref with binding."""
        expr = Ref(key="x")
        bindings = {"x": Ref(key="y")}
        result = substitute(expr, bindings)
        assert isinstance(result, Ref)
        assert result.key == "y"

    def test_substitute_ref_passthrough(self):
        """Substitute keeps Ref if no binding."""
        expr = Ref(key="x")
        bindings = {}
        result = substitute(expr, bindings)
        assert result == expr

    def test_substitute_slice(self):
        """Substitute recurses into Slice."""
        expr = Slice(expr=Ref(key="x"), slices=((0, 5, None),))
        bindings = {"x": Ref(key="y")}
        result = substitute(expr, bindings)
        assert isinstance(result, Slice)
        assert isinstance(result.expr, Ref)
        assert result.expr.key == "y"

    def test_substitute_concat(self):
        """Substitute recurses into Concat children."""
        expr = Concat(exprs=(Ref(key="a"), Ref(key="b")), dim=0)
        bindings = {"a": Ref(key="x"), "b": Ref(key="y")}
        result = substitute(expr, bindings)
        assert isinstance(result, Concat)
        assert result.exprs[0].key == "x"
        assert result.exprs[1].key == "y"

    def test_substitute_init_unchanged(self):
        """Substitute leaves Init unchanged."""
        expr = Init(shape=(10,), init_type="zeros")
        result = substitute(expr, {"x": Ref(key="y")})
        assert result == expr

    def test_substitute_complex(self):
        """Substitute handles complex nested expressions."""
        # Concat of Slice(Ref) and Init
        expr = Concat(exprs=(
            Slice(expr=Ref(key="a"), slices=((0, 5, None),)),
            Init(shape=(5,), init_type="zeros"),
        ), dim=0)
        bindings = {"a": Ref(key="source")}
        result = substitute(expr, bindings)

        assert isinstance(result, Concat)
        assert isinstance(result.exprs[0], Slice)
        assert result.exprs[0].expr.key == "source"
        assert isinstance(result.exprs[1], Init)


class TestFuse:
    """Test expression fusion/optimization."""

    def test_fuse_flatten_concat(self):
        """Fuse flattens nested Concat with same dim."""
        inner = Concat(exprs=(Ref(key="a"), Ref(key="b")), dim=0)
        outer = Concat(exprs=(inner, Ref(key="c"),), dim=0)
        result = fuse(outer)

        assert isinstance(result, Concat)
        assert len(result.exprs) == 3
        assert result.exprs[0].key == "a"
        assert result.exprs[1].key == "b"
        assert result.exprs[2].key == "c"

    def test_fuse_no_flatten_different_dim(self):
        """Fuse doesn't flatten Concat with different dim."""
        inner = Concat(exprs=(Ref(key="a"), Ref(key="b")), dim=1)
        outer = Concat(exprs=(inner, Ref(key="c"),), dim=0)
        result = fuse(outer)

        assert isinstance(result, Concat)
        assert len(result.exprs) == 2
        assert isinstance(result.exprs[0], Concat)

    def test_fuse_reshape_reshape(self):
        """Fuse collapses nested Reshape."""
        expr = Reshape(expr=Reshape(expr=Ref(key="a"), shape=(4, 5)), shape=(2, 10))
        result = fuse(expr)

        assert isinstance(result, Reshape)
        assert result.shape == (2, 10)
        assert isinstance(result.expr, Ref)


class TestSerialization:
    """Test expression and plan serialization."""

    def test_ref_roundtrip(self):
        """Ref serializes and deserializes."""
        expr = Ref(key="model.weight")
        d = expr.model_dump()
        restored = ExprAdapter.validate_python(d)
        assert isinstance(restored, Ref)
        assert restored.key == expr.key

    def test_slice_roundtrip(self):
        """Slice serializes and deserializes."""
        expr = Slice(expr=Ref(key="a"), slices=((0, 5, None), (None, None, 2)))
        d = expr.model_dump()
        restored = ExprAdapter.validate_python(d)
        assert isinstance(restored, Slice)
        assert restored.slices == expr.slices

    def test_concat_roundtrip(self):
        """Concat serializes and deserializes."""
        expr = Concat(exprs=(Ref(key="a"), Init(shape=(5,), init_type="zeros")), dim=1)
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
        expr = Reshape(expr=Ref(key="a"), shape=(4, 5))
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
                "out.x": Ref(key="in.x"),
                "out.y": Concat(exprs=(Ref(key="in.a"), Init(shape=(5,), init_type="zeros")), dim=0),
            },
        )

        d = plan.model_dump()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        restored = ExprPlan.model_validate(d2)

        assert len(restored) == 2
        assert restored.source_format == "a"
        assert restored.target_format == "b"
        assert "out.x" in restored
        assert "out.y" in restored


class TestExprPlan:
    """Test ExprPlan class."""

    def test_plan_define_and_access(self):
        """Plan stores and retrieves expressions."""
        plan = ExprPlan(mappings={
            "target": Ref(key="source"),
        })
        assert "target" in plan
        assert isinstance(plan["target"], Ref)

    def test_plan_source_keys(self):
        """Plan identifies all source references."""
        plan = ExprPlan(mappings={
            "a": Ref(key="x"),
            "b": Concat(exprs=(Ref(key="y"), Ref(key="z")), dim=0),
            "c": Init(shape=(10,), init_type="zeros"),
        })

        assert plan.source_keys() == {"x", "y", "z"}

    def test_plan_target_keys(self):
        """Plan identifies all target keys."""
        plan = ExprPlan(mappings={
            "a": Ref(key="x"),
            "b": Ref(key="y"),
        })

        assert plan.target_keys() == {"a", "b"}

    def test_plan_summary(self):
        """Plan summary provides useful info."""
        plan = ExprPlan(
            source_format="llava",
            target_format="apriel2",
            mappings={
                "a": Ref(key="x"),
                "b": Concat(exprs=(Ref(key="y"), Ref(key="z")), dim=0),
                "c": Init(shape=(10,), init_type="zeros"),
            },
        )

        summary = plan.summary()
        assert summary["source_format"] == "llava"
        assert summary["target_format"] == "apriel2"
        assert summary["num_targets"] == 3
        assert summary["num_source_refs"] == 3

    def test_plan_fuse(self):
        """Plan fuse applies optimizations."""
        inner = Concat(exprs=(Ref(key="a"), Ref(key="b")), dim=0)
        plan = ExprPlan(mappings={
            "out": Concat(exprs=(inner, Ref(key="c"),), dim=0),
        })

        fused = plan.fuse()
        assert isinstance(fused["out"], Concat)
        assert len(fused["out"].exprs) == 3


class TestComposition:
    """Test plan composition."""

    def test_compose_simple_refs(self):
        """Compose simple Ref chains."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                "intermediate": Ref(key="original"),
            },
        )

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                "final": Ref(key="intermediate"),
            },
        )

        composed = plan1 | plan2

        assert composed.source_format == "a"
        assert composed.target_format == "c"
        assert "final" in composed
        assert isinstance(composed["final"], Ref)
        assert composed["final"].key == "original"

    def test_compose_with_concat(self):
        """Compose through Concat expressions."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                "x": Ref(key="src_x"),
                "y": Ref(key="src_y"),
            },
        )

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                "combined": Concat(exprs=(Ref(key="x"), Ref(key="y")), dim=0),
            },
        )

        composed = plan1 | plan2

        assert "combined" in composed
        result = composed["combined"]
        assert isinstance(result, Concat)
        assert result.exprs[0].key == "src_x"
        assert result.exprs[1].key == "src_y"

    def test_compose_with_slice(self):
        """Compose through Slice expressions."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                "full": Ref(key="source"),
            },
        )

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                "partial": Slice(expr=Ref(key="full"), slices=((0, 5, None),)),
            },
        )

        composed = plan1 | plan2

        result = composed["partial"]
        assert isinstance(result, Slice)
        assert isinstance(result.expr, Ref)
        assert result.expr.key == "source"

    def test_compose_preserves_init(self):
        """Compose preserves Init expressions."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                "x": Ref(key="src"),
            },
        )

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                "combined": Concat(exprs=(Ref(key="x"), Init(shape=(5,), init_type="zeros")), dim=0),
            },
        )

        composed = plan1 | plan2

        result = composed["combined"]
        assert isinstance(result.exprs[0], Ref)
        assert result.exprs[0].key == "src"
        assert isinstance(result.exprs[1], Init)

    def test_compose_passthrough(self):
        """Compose keeps refs that plan1 doesn't produce."""
        plan1 = ExprPlan(
            source_format="a",
            target_format="b",
            mappings={
                "x": Ref(key="src_x"),
            },
        )
        # plan1 doesn't define "passthrough"

        plan2 = ExprPlan(
            source_format="b",
            target_format="c",
            mappings={
                "out": Concat(exprs=(Ref(key="x"), Ref(key="passthrough")), dim=0),
            },
        )

        composed = plan1 | plan2

        result = composed["out"]
        assert result.exprs[0].key == "src_x"  # Substituted
        assert result.exprs[1].key == "passthrough"  # Kept as-is


class TestStreamingExecution:
    """Test streaming execution with ref-counting."""

    def test_execute_simple(self):
        """Execute simple plan."""
        plan = ExprPlan(mappings={
            "out": Ref(key="in"),
        })

        sources = {"in": torch.tensor([1.0, 2.0, 3.0])}
        result = execute(plan, sources)

        assert "out" in result
        assert torch.allclose(result["out"], sources["in"])

    def test_execute_concat(self):
        """Execute plan with Concat."""
        plan = ExprPlan(mappings={
            "combined": Concat(exprs=(Ref(key="a"), Ref(key="b")), dim=0),
        })

        sources = {
            "a": torch.ones(2, 3),
            "b": torch.zeros(3, 3),
        }
        result = execute(plan, sources)

        assert result["combined"].shape == (5, 3)

    def test_execute_mil_like(self):
        """Execute MIL-like Concat of Slices and Init."""
        # Simulated MIL: in_proj = [z, x, B, C]
        plan = ExprPlan(mappings={
            "in_proj": Concat(exprs=(
                Init(shape=(4, 8), init_type="zeros"),  # z
                Slice(expr=Ref(key="v"), slices=((0, 2, None), (None, None, None))),  # x
                Slice(expr=Ref(key="k"), slices=((0, 2, None), (None, None, None))),  # B
                Slice(expr=Ref(key="q"), slices=((0, 4, None), (None, None, None))),  # C
            ), dim=0),
        })

        sources = {
            "q": torch.ones(4, 8),
            "k": torch.full((2, 8), 2.0),
            "v": torch.full((2, 8), 3.0),
        }
        result = execute(plan, sources)

        assert result["in_proj"].shape == (12, 8)
        assert torch.allclose(result["in_proj"][0:4], torch.zeros(4, 8))  # z
        assert torch.allclose(result["in_proj"][4:6], torch.full((2, 8), 3.0))  # x <- v
        assert torch.allclose(result["in_proj"][6:8], torch.full((2, 8), 2.0))  # B <- k
        assert torch.allclose(result["in_proj"][8:12], torch.ones(4, 8))  # C <- q

    def test_streaming_ref_counting(self):
        """Streaming executor releases sources after use."""
        plan = ExprPlan(mappings={
            "out1": Ref(key="shared"),
            "out2": Ref(key="shared"),
            "out3": Ref(key="unique"),
        })

        load_calls = []

        def loader(key: str) -> torch.Tensor:
            load_calls.append(key)
            return torch.randn(10)

        executor = StreamingExecutor(plan, loader)

        # Consume all results
        results = list(executor.execute())

        # Each source should be loaded exactly once
        assert load_calls.count("shared") == 1
        assert load_calls.count("unique") == 1
        assert len(results) == 3

    def test_streaming_memory_cleanup(self):
        """Streaming executor cleans up memory."""
        plan = ExprPlan(mappings={
            "out": Ref(key="in"),
        })

        cache_state = {"loaded": False, "released": False}

        class TrackedTensor:
            def __init__(self):
                cache_state["loaded"] = True

            def clone(self):
                return torch.randn(10)

            def to(self, **kwargs):
                return self

        def loader(key: str):
            return TrackedTensor()

        executor = StreamingExecutor(plan, loader)
        list(executor.execute())  # Consume all

        # Executor should complete without assertion error (cache empty)


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
        )

        # Check in_proj is Concat
        in_proj = exprs["model.decoder.blocks.0.mixer.in_proj.weight"]
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
        out_proj = exprs["model.decoder.blocks.0.mixer.out_proj.weight"]
        assert isinstance(out_proj, Ref)

    def test_plan_mil_execution(self):
        """MIL plan executes correctly with actual weights."""
        exprs = plan_mil_attention_to_mamba(
            layer_idx=0,
            hidden_size=64,
            d_inner=128,
            d_xb=32,
            dt_rank=4,
            d_state=16,
            source_prefix="attn.",
            target_prefix="mamba.",
        )

        # Build mappings dict from exprs
        mappings = {}
        for key, expr in exprs.items():
            # Adjust keys for test
            adjusted_key = key.replace("model.decoder.blocks.0.mixer.", "")
            mappings[adjusted_key] = expr

        plan = ExprPlan(mappings=mappings)

        # Create attention weights
        sources = {
            "attn.q_proj.weight": torch.full((128, 64), 1.0),
            "attn.k_proj.weight": torch.full((32, 64), 2.0),
            "attn.v_proj.weight": torch.full((32, 64), 3.0),
            "attn.o_proj.weight": torch.full((64, 128), 4.0),
        }

        result = execute(plan, sources)

        # Verify in_proj layout: [z, x, B, C]
        in_proj = result["mamba.in_proj.weight"]
        assert in_proj.shape == (128 + 32 + 32 + 128, 64)

        # z (0:128) is random init
        # x (128:160) should be 3.0 (from v)
        assert torch.allclose(in_proj[128:160], torch.full((32, 64), 3.0))
        # B (160:192) should be 2.0 (from k)
        assert torch.allclose(in_proj[160:192], torch.full((32, 64), 2.0))
        # C (192:320) should be 1.0 (from q)
        assert torch.allclose(in_proj[192:320], torch.full((128, 64), 1.0))

        # out_proj should be 4.0
        assert torch.allclose(result["mamba.out_proj.weight"], torch.full((64, 128), 4.0))


class TestFullPipeline:
    """Test full conversion + surgery pipeline."""

    def test_compose_llava_to_mamba(self, llava_pixtral_config, apriel2_config_stochastic):
        """Can compose Llava conversion with surgery to stochastic."""
        # Build conversion plan
        conversion_plan = plan_llava_to_apriel2(llava_pixtral_config)

        # Build surgery plan (need intermediate config)
        from fast_llm_external_models.apriel2.convert_from_llava import convert_config
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
        result = execute(conversion_plan, source_weights)

        assert len(result) > 0

        # Verify key mappings worked
        assert "model.embed_tokens.weight" in result
        assert any("mixer.self_attn" in k for k in result)


class TestExpressionRepr:
    """Test expression string representations."""

    def test_ref_repr(self):
        """Ref has readable repr."""
        expr = Ref(key="model.weight")
        assert "model.weight" in repr(expr)

    def test_slice_repr(self):
        """Slice has readable repr."""
        expr = Slice(expr=Ref(key="a"), slices=((0, 5, None), (None, None, None)))
        r = repr(expr)
        # Repr shows :5 for 0:5 (standard Python slice notation)
        assert ":5" in r
        assert ":" in r

    def test_concat_repr(self):
        """Concat has readable repr."""
        expr = Concat(exprs=(Ref(key="a"), Ref(key="b")), dim=0)
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
                        "type": "gated_delta_net",
                        "init": "transfer",  # explicitly request transfer
                        "num_value_heads": 4,
                        "num_key_heads": 2,
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
                        "type": "gated_delta_net",
                        "init": "random",  # random init - no converter needed
                        "num_value_heads": 4,
                        "num_key_heads": 2,
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
