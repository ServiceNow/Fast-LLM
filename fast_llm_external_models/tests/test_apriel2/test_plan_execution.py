"""test_plan_execution.py - Plan execution and algebraic composition laws.

This module provides rigorous, parameterized tests for the mathematical properties
that the conversion system must satisfy. Each test class corresponds to one
algebraic structure, and each test method verifies one specific law.

Conceptual Types
================

The conversion system operates on three conceptual types (all ``dict`` at runtime):

- **S (State)**: Complete config without ``init`` fields
- **P (Partial Surgery)**: Incomplete config, may have ``init`` fields
- **T (Transition Spec)**: Complete config WITH ``init`` fields

Algebraic Structures
====================

1. **Partial Surgeries (P)** form a **Monoid** under deep merge::

       compose_configs : P × P → P
       Identity: {}
       Associativity: (p1 ∘ p2) ∘ p3 = p1 ∘ (p2 ∘ p3)

2. **Surgeries act on States** to produce Transition Specs::

       compose_configs : S × P → T
       compose_configs : T × P → T

   Action law (additive surgeries): (s · p1) · p2 = s · (p1 ∘ p2)

3. **Plans** form a **Category** with composition::

       compose : Plan(A→B) × Plan(B→C) → Plan(A→C)
       Associativity: (P1 ∘ P2) ∘ P3 = P1 ∘ (P2 ∘ P3)

4. **plan_surgery is a Functor** from config pairs to plans::

       plan_surgery : S × T → Plan
       Functoriality: compose(plan(S,T1), plan(T1,T2)) ≡ plan(S,T2)

   This is semantic equivalence: both produce identical weights when executed.

Important Behaviors Tested
==========================

- **init stripping**: Between surgery iterations, T → S conversion via
  ``strip_init_fields()`` ensures ``init: random`` applies only to the surgery
  that introduces a component.

- **Bias inheritance**: Per-layer bias settings propagate through surgery chains.

- **Plan composition**: Composed plans produce identical weights to direct plans.

Design Principles
=================

- Each law gets ONE parameterized test, not multiple similar tests
- Fixtures provide diverse configs (with/without biases)
- Corner cases are covered via parameterization, not test proliferation
- Tests document the laws they verify in their docstrings
"""

import pytest
import torch
from functools import reduce

from fast_llm_external_models.apriel2.conversion import (
    compose,
    compose_configs,
    execute,
    plan_surgery,
    ExprPlan,
    W,
    Ref,
    Concat,
    Slice,
    Init,
)

# Import shared helper from conftest
from fast_llm_external_models.tests.test_apriel2.conftest import make_weights_for_config


# =============================================================================
# Fixtures: Use shared fixtures from conftest.py where possible
# =============================================================================
# - base_config_dict: Complete config without biases (Llama-style)
# - base_config_with_bias_dict: Complete config with QKV biases
# - additive_surgery_chain: [wrap_stochastic, add_sliding_window, add_gdn]
# =============================================================================


# =============================================================================
# Test: Plan Composition Associativity
# =============================================================================


class TestPlanCompositionAssociativity:
    """
    LAW: Plan composition is associative.

        (P₁ ∘ P₂) ∘ P₃ = P₁ ∘ (P₂ ∘ P₃)

    where ∘ denotes compose(P1, P2).

    This must hold for the AST structure, not just semantic equivalence.
    """

    @pytest.mark.parametrize("expr_type", ["ref_chain", "with_concat", "with_slice", "with_init"])
    def test_associativity(self, expr_type):
        """Plan composition is associative for various expression types."""
        # Build three plans that can be composed
        if expr_type == "ref_chain":
            p1 = ExprPlan(mappings={W("b"): Ref(key=W("a"))})
            p2 = ExprPlan(mappings={W("c"): Ref(key=W("b"))})
            p3 = ExprPlan(mappings={W("d"): Ref(key=W("c"))})
        elif expr_type == "with_concat":
            p1 = ExprPlan(mappings={W("x"): Ref(key=W("a")), W("y"): Ref(key=W("b"))})
            p2 = ExprPlan(mappings={W("xy"): Concat(exprs=(Ref(key=W("x")), Ref(key=W("y"))), dim=0)})
            p3 = ExprPlan(mappings={W("final"): Ref(key=W("xy"))})
        elif expr_type == "with_slice":
            p1 = ExprPlan(mappings={W("full"): Ref(key=W("src"))})
            p2 = ExprPlan(mappings={W("part"): Slice(expr=Ref(key=W("full")), slices=((0, 5, None),))})
            p3 = ExprPlan(mappings={W("out"): Ref(key=W("part"))})
        elif expr_type == "with_init":
            p1 = ExprPlan(mappings={W("x"): Ref(key=W("a"))})
            p2 = ExprPlan(mappings={W("y"): Concat(exprs=(Ref(key=W("x")), Init(shape=(5,), init_type="zeros")), dim=0)})
            p3 = ExprPlan(mappings={W("z"): Ref(key=W("y"))})

        left = compose(compose(p1, p2), p3)
        right = compose(p1, compose(p2, p3))

        assert left.mappings == right.mappings, f"Associativity failed for {expr_type}"


# =============================================================================
# Test: Functoriality of plan_surgery (THE CRITICAL PROPERTY)
# =============================================================================


class TestPlanSurgeryFunctoriality:
    """
    LAW: plan_surgery is functorial with respect to config composition.

    For a surgery chain P₁, P₂, ..., Pₙ applied to base state S₀::

        T₁ = compose_configs(S₀, P₁)        # S × P → T
        T₂ = compose_configs(T₁, P₂)        # T × P → T (no stripping!)
        ...
        Tₙ = compose_configs(Tₙ₋₁, Pₙ)

    Plan functoriality says::

        compose(plan(S₀,T₁), plan(T₁,T₂), ...) ≡ plan(S₀, Tₙ)

    where ≡ denotes semantic equivalence (identical weights when executed).

    NOTE: This tests T × P composition WITHOUT stripping between steps.
    This differs from build_plan which strips (T → S) between iterations.
    Both patterns are valid:

    - Without stripping: init fields accumulate, testing plan composition purity
    - With stripping: init consumed per-step, testing real usage (see
      test_build_plan_strips_init_between_iterations)

    The functoriality law holds in both cases because plan composition
    correctly substitutes Ref expressions with their definitions.
    """

    @pytest.mark.parametrize("chain_length", [1, 2, 3])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_functoriality(
        self,
        chain_length,
        use_bias,
        base_config_dict,
        base_config_with_bias_dict,
        additive_surgery_chain,
    ):
        """
        Composed incremental plans produce same weights as direct plan.

        Parameterized over:
        - chain_length: Number of surgeries (1, 2, or 3)
        - use_bias: Whether base config has biases
        """
        base_config = base_config_with_bias_dict if use_bias else base_config_dict
        surgeries = additive_surgery_chain[:chain_length]

        # Build config chain: C₀ → C₁ → ... → Cₙ
        configs = [base_config]
        for s in surgeries:
            configs.append(compose_configs(configs[-1], s))

        # Build incremental plans: Pₖ = plan_surgery(Cₖ₋₁, Cₖ)
        plans = [plan_surgery(configs[i], configs[i+1]) for i in range(len(surgeries))]

        # Compose all incremental plans
        composed_plan = reduce(compose, plans)

        # Build direct plan: plan_surgery(C₀, Cₙ)
        direct_plan = plan_surgery(configs[0], configs[-1])

        # Execute both on same weights
        weights = make_weights_for_config(base_config)
        composed_weights = execute(composed_plan, weights, seed=42)
        direct_weights = execute(direct_plan, weights, seed=42)

        # Verify semantic equivalence
        assert set(composed_weights.keys()) == set(direct_weights.keys()), \
            f"Key sets differ for chain_length={chain_length}, use_bias={use_bias}"

        for key in composed_weights:
            assert torch.allclose(composed_weights[key], direct_weights[key], atol=1e-6), \
                f"Weight mismatch for {key} with chain_length={chain_length}, use_bias={use_bias}"

    @pytest.mark.parametrize("split_point", [1, 2])
    def test_arbitrary_grouping(
        self,
        split_point,
        base_config_with_bias_dict,
        additive_surgery_chain,
    ):
        """
        Any grouping of surgery chain produces same result.

        For surgeries [S₁, S₂, S₃], tests that:
        - compose(P₁, compose(P₂, P₃))
        - compose(compose(P₁, P₂), P₃)
        - plan_surgery(C₀, C₃)

        all produce identical weights.
        """
        surgeries = additive_surgery_chain

        # Build config chain
        configs = [base_config_with_bias_dict]
        for s in surgeries:
            configs.append(compose_configs(configs[-1], s))

        # Build incremental plans
        plans = [plan_surgery(configs[i], configs[i+1]) for i in range(3)]

        # Different groupings
        left_grouped = compose(compose(plans[0], plans[1]), plans[2])
        right_grouped = compose(plans[0], compose(plans[1], plans[2]))
        direct = plan_surgery(configs[0], configs[-1])

        # Execute all
        weights = make_weights_for_config(base_config_with_bias_dict)
        results = {
            "left": execute(left_grouped, weights, seed=42),
            "right": execute(right_grouped, weights, seed=42),
            "direct": execute(direct, weights, seed=42),
        }

        # All must match
        keys = set(results["left"].keys())
        assert keys == set(results["right"].keys()) == set(results["direct"].keys())

        for key in keys:
            assert torch.allclose(results["left"][key], results["right"][key], atol=1e-6)
            assert torch.allclose(results["left"][key], results["direct"][key], atol=1e-6)


# =============================================================================
# Test: Bias Inheritance Preservation (Regression for the specific bug)
# =============================================================================


class TestBiasInheritancePreservation:
    """
    PROPERTY: Per-layer bias settings must be preserved through surgery chains.

    When a surgery spec does not mention bias settings, they must be inherited
    from the source config. This is the specific failure mode of the build_plan
    bug: passing partial surgery specs to plan_surgery lost inherited fields.

    This test verifies the SYMPTOM (missing biases) rather than the LAW
    (functoriality). It's kept as a focused regression test.
    """

    @pytest.mark.parametrize("num_surgeries", [1, 2, 3])
    def test_qkv_biases_preserved_through_chain(
        self,
        num_surgeries,
        base_config_with_bias_dict,
        additive_surgery_chain,
    ):
        """QKV biases (enabled in source) appear in plan after N surgeries."""
        surgeries = additive_surgery_chain[:num_surgeries]

        # Build config and plan chain
        configs = [base_config_with_bias_dict]
        for s in surgeries:
            configs.append(compose_configs(configs[-1], s))

        plans = [plan_surgery(configs[i], configs[i+1]) for i in range(num_surgeries)]
        final_plan = reduce(compose, plans) if len(plans) > 1 else plans[0]

        # Check bias keys present
        target_keys = {str(k) for k in final_plan.target_keys()}

        assert any("q_proj.bias" in k for k in target_keys), \
            f"q_proj.bias missing after {num_surgeries} surgeries"
        assert any("k_proj.bias" in k for k in target_keys), \
            f"k_proj.bias missing after {num_surgeries} surgeries"
        assert any("v_proj.bias" in k for k in target_keys), \
            f"v_proj.bias missing after {num_surgeries} surgeries"
        # O bias should NOT be present (disabled in source)
        assert not any("o_proj.bias" in k for k in target_keys), \
            f"o_proj.bias should not be present (disabled in source)"

    def test_bias_values_preserved(
        self,
        base_config_with_bias_dict,
        additive_surgery_chain,
    ):
        """Bias tensor values are correctly transferred, not just keys."""
        surgery = additive_surgery_chain[0]  # wrap_stochastic
        c1 = compose_configs(base_config_with_bias_dict, surgery)
        plan = plan_surgery(base_config_with_bias_dict, c1)

        weights = make_weights_for_config(base_config_with_bias_dict)
        result = execute(plan, weights, seed=42)

        # Verify values match (not just that keys exist)
        for i in range(base_config_with_bias_dict["decoder"]["num_blocks"]):
            src_key = W(f"model.decoder.blocks.{i}.mixer.q_proj.bias")
            dst_key = W(f"model.decoder.blocks.{i}.mixer.mixers.attention.q_proj.bias")

            assert dst_key in result, f"Missing {dst_key}"
            assert torch.allclose(weights[src_key], result[dst_key]), \
                f"Bias values differ for block {i}"


# =============================================================================
# Test: build_plan Integration (Regression test for convert.py)
# =============================================================================


class TestBuildPlanIntegration:
    """
    REGRESSION: build_plan must compose configs before calling plan_surgery.

    The bug was:
        plan_surgery(current_config, surgery_config)  # WRONG: partial

    Should be:
        target = compose_configs(current_config, surgery_config)
        plan_surgery(current_config, target)  # CORRECT: complete

    This test verifies the fix in convert.py's build_plan function.
    """

    @pytest.mark.parametrize("num_surgeries", [1, 2])
    def test_build_plan_preserves_inherited_fields(
        self,
        num_surgeries,
        base_config_with_bias_dict,
        additive_surgery_chain,
    ):
        """build_plan produces plans with inherited bias mappings."""
        from fast_llm_external_models.apriel2.convert import build_plan

        surgeries = additive_surgery_chain[:num_surgeries]

        plan, final_config = build_plan(
            base_config_with_bias_dict,
            surgeries,
            source_format="apriel2",
        )

        # Verify inherited biases in config
        if num_surgeries >= 1:
            attn = final_config["decoder"]["block"]["mixer"]["mixers"]["attention"]
            assert attn.get("query_layer", {}).get("bias", {}).get("enabled") is True

        # Verify bias mappings in plan
        target_keys = {str(k) for k in plan.target_keys()}
        assert any("q_proj.bias" in k for k in target_keys), \
            f"build_plan with {num_surgeries} surgeries missing q_proj.bias"


# =============================================================================
# Test: init Field Preservation (Critical for random initialization)
# =============================================================================


class TestInitFieldPreservation:
    """
    PROPERTY: The `init` field must be visible to plan_surgery.

    The `init` field controls weight initialization mode:
    - `init: transfer` → use weight transfer/conversion
    - `init: random` → use random initialization

    compose_configs must preserve `init` so plan_surgery can see it.
    Stripping happens only at final output (when saving to disk).
    """

    def test_init_random_produces_init_expression(self, base_config_with_bias_dict):
        """Surgery with init: random produces Init expressions in plan."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "gdn": {"type": "gdn", "init": "random", "convolution_layer": {"kernel_size": 4}},
                        },
                    },
                },
            },
        }

        target = compose_configs(base_config_with_bias_dict, surgery)
        plan = plan_surgery(base_config_with_bias_dict, target)

        # Check that GDN weights use Init expressions (random init)
        target_keys = {str(k) for k in plan.target_keys()}
        gdn_keys = [k for k in target_keys if "gdn" in k.lower()]

        assert len(gdn_keys) > 0, "No GDN keys in plan"

        # Verify at least one GDN weight uses Init (random initialization)
        has_init_expr = False
        for key in plan.target_keys():
            if "gdn" in str(key).lower():
                expr = plan.mappings[key]
                if isinstance(expr, Init):
                    has_init_expr = True
                    break
                # Also check inside Concat/other composite expressions
                if hasattr(expr, 'exprs'):
                    for sub in expr.exprs:
                        if isinstance(sub, Init):
                            has_init_expr = True
                            break

        assert has_init_expr, "init: random should produce Init expressions for GDN weights"

    def test_init_transfer_produces_ref_expression(self, base_config_with_bias_dict):
        """Surgery with init: transfer produces Ref expressions (weight transfer)."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                        },
                    },
                },
            },
        }

        target = compose_configs(base_config_with_bias_dict, surgery)
        plan = plan_surgery(base_config_with_bias_dict, target)

        # Check that attention weights use Ref expressions (transfer)
        has_ref = False
        for key in plan.target_keys():
            if "attention" in str(key) and "q_proj.weight" in str(key):
                expr = plan.mappings[key]
                if isinstance(expr, Ref):
                    has_ref = True
                    break

        assert has_ref, "init: transfer should produce Ref expressions for attention weights"

    def test_build_plan_respects_init_random(self, base_config_with_bias_dict):
        """build_plan correctly uses init: random for weight initialization."""
        from fast_llm_external_models.apriel2.convert import build_plan

        # Mamba requires many config fields for random init
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "mamba": {
                                "type": "mamba",
                                "init": "random",
                                "d_inner": 512,
                                "d_state": 16,
                                "dt_rank": 16,
                                "d_xb": 64,
                                "d_conv": 4,
                                "repeat_kv_before_conv": False,
                                "conv_bias": True,
                                "dt_proj_bias": True,
                                "dt_min": 0.001,
                                "dt_max": 0.1,
                                "dt_init_floor": 1e-4,
                            },
                        },
                    },
                },
            },
        }

        plan, final_config = build_plan(
            base_config_with_bias_dict,
            [surgery],
            source_format="apriel2",
        )

        # Verify mamba weights use Init (random init)
        has_mamba_init = False
        for key in plan.target_keys():
            key_str = str(key)
            if "mamba" in key_str:
                expr = plan.mappings[key]
                if isinstance(expr, Init):
                    has_mamba_init = True
                    break

        assert has_mamba_init, "build_plan should use Init for init: random components"

    def test_build_plan_strips_init_between_iterations(self, base_config_with_bias_dict):
        """build_plan strips init between iterations (T → S conversion).

        This tests that the intermediate state between surgeries has no init fields.
        The composed plan will show Init expressions because plan composition
        substitutes Ref → Init, but the semantics are correct: GDN is initialized
        once (in surgery 1), not re-randomized in surgery 2.
        """
        from fast_llm_external_models.apriel2.conversion import (
            compose_configs, strip_init_fields, plan_surgery, compose
        )

        # Surgery 1: Add GDN with random init
        surgery1 = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"init": "transfer"},
                            "gdn": {
                                "type": "gdn",
                                "init": "random",
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                },
            },
        }

        # Surgery 2: Add sliding window (doesn't mention GDN)
        surgery2 = {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "sliding_window": {"init": "transfer", "window_size": 512},
                        },
                    },
                },
            },
        }

        # Simulate build_plan's iteration loop
        s0 = base_config_with_bias_dict

        # Iteration 1
        t1 = compose_configs(s0, surgery1)
        assert t1["decoder"]["block"]["mixer"]["mixers"]["gdn"].get("init") == "random"
        s1 = strip_init_fields(t1)
        assert s1["decoder"]["block"]["mixer"]["mixers"]["gdn"].get("init") is None

        # Iteration 2: s1 has no init for GDN
        t2 = compose_configs(s1, surgery2)
        assert t2["decoder"]["block"]["mixer"]["mixers"]["gdn"].get("init") is None, \
            "GDN should have no init in T2 (wasn't in surgery2, stripped from s1)"

        # plan_surgery(s1, t2) should use Ref for GDN (transfer, not random)
        plan2 = plan_surgery(s1, t2)
        gdn_uses_ref = False
        for key in plan2.target_keys():
            if "gdn" in str(key):
                expr = plan2.mappings[key]
                if isinstance(expr, Ref):
                    gdn_uses_ref = True
                    break

        assert gdn_uses_ref, "plan_surgery(s1, t2) should use Ref for GDN (transfer from s1)"
