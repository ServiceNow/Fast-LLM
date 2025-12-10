"""End-to-end torture test for plan composition.

This tests the FULL pipeline at every step of a surgery chain:
1. Config composition produces valid configs
2. Plan building works for each surgery
3. Plan execution produces valid weights
4. Models can be instantiated with the weights
5. Forward pass works

This is the ultimate integration test for the conversion system.
"""

import json
from pathlib import Path

import pytest
import torch

from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config
from fast_llm_external_models.apriel2.conversion import compose, compose_configs, execute, plan_surgery
from fast_llm_external_models.apriel2.conversion.llava import convert_config as convert_llava_config
from fast_llm_external_models.apriel2.conversion.llava import plan_llava_to_apriel2
from fast_llm_external_models.apriel2.modeling_apriel2 import Apriel2ForConditionalGeneration
from fast_llm_external_models.tests.test_apriel2.conftest import requires_cuda

# =============================================================================
# Cycling Surgery Generation
# =============================================================================


def get_stochastic_blocks(config: dict) -> dict[str, dict]:
    """Extract all stochastic blocks from a config.

    Returns:
        Dict mapping block_path -> mixer_config for all stochastic mixers.
        For fixed decoder: {"block": mixer_config}
        For pattern decoder: {"blocks.name": mixer_config, ...}
    """
    decoder = config.get("decoder", {})
    decoder_type = decoder.get("type", "fixed")

    stochastic_blocks = {}

    if decoder_type == "fixed":
        block = decoder.get("block", {})
        mixer = block.get("mixer", {})
        if mixer.get("type") == "stochastic":
            stochastic_blocks["block"] = mixer
    else:  # pattern
        blocks = decoder.get("blocks", {})
        for block_name, block in blocks.items():
            mixer = block.get("mixer", {})
            if mixer.get("type") == "stochastic":
                stochastic_blocks[f"blocks.{block_name}"] = mixer

    return stochastic_blocks


def generate_cycling_surgeries(config: dict) -> list[tuple[dict, str]]:
    """Generate cycling surgeries to test all sub-mixers in stochastic blocks.

    For each stochastic block, generates surgeries to cycle through all
    sub-mixers that aren't the main mixer, then restores the original main.

    Returns:
        List of (surgery, description) tuples. The last surgery for each block
        restores the original main_mixer_name.
    """
    stochastic_blocks = get_stochastic_blocks(config)
    surgeries = []

    for block_path, mixer in stochastic_blocks.items():
        main_mixer = mixer.get("main_mixer_name", "attention")
        sub_mixer_names = list(mixer.get("mixers", {}).keys())

        # Generate cycling surgeries for non-main mixers
        for sub_name in sub_mixer_names:
            if sub_name != main_mixer:
                # Build surgery path based on block_path
                if block_path == "block":
                    surgery = {"decoder": {"block": {"mixer": {"main_mixer_name": sub_name}}}}
                else:
                    # block_path is "blocks.block_name"
                    block_name = block_path.split(".")[1]
                    surgery = {"decoder": {"blocks": {block_name: {"mixer": {"main_mixer_name": sub_name}}}}}
                surgeries.append((surgery, f"cycle {block_path} to {sub_name}"))

        # Restore original main_mixer_name
        if any(sub_name != main_mixer for sub_name in sub_mixer_names):
            if block_path == "block":
                restore = {"decoder": {"block": {"mixer": {"main_mixer_name": main_mixer}}}}
            else:
                block_name = block_path.split(".")[1]
                restore = {"decoder": {"blocks": {block_name: {"mixer": {"main_mixer_name": main_mixer}}}}}
            surgeries.append((restore, f"restore {block_path} to {main_mixer}"))

    return surgeries


def expand_surgery_chain_with_cycling(
    surgery_chain: list[dict],
    initial_config: dict,
) -> list[tuple[dict, str, bool]]:
    """Expand a surgery chain with cycling surgeries.

    After each surgery that produces stochastic mixers, inserts cycling surgeries
    to test all sub-mixers, then restores the original main_mixer_name.

    Args:
        surgery_chain: Original surgery chain.
        initial_config: Config before applying any surgeries.

    Returns:
        Expanded list of (surgery, description, is_restore) tuples.
        is_restore=True for restore surgeries (forward pass is redundant but validates state).
    """
    expanded = []
    current_config = initial_config

    for i, surgery in enumerate(surgery_chain):
        # Add the original surgery
        expanded.append((surgery, f"surgery {i+1}", False))

        # Apply surgery to get new config
        current_config = compose_configs(current_config, surgery)

        # Generate cycling surgeries for any stochastic blocks
        cycling = generate_cycling_surgeries(current_config)

        for cycling_surgery, desc in cycling:
            is_restore = desc.startswith("restore")
            expanded.append((cycling_surgery, desc, is_restore))

            # Apply cycling surgery (for next iteration's context)
            # Note: restore brings us back to post-original-surgery state
            current_config = compose_configs(current_config, cycling_surgery)

    return expanded


class TestPlanCompositionTorture:
    """End-to-end torture test for plan composition.

    Tests that the FULL system works at every step of a complex surgery chain:
    - Llava → Apriel2 (initial conversion)
    - Then a chain of surgeries adding/modifying mixers

    At each step, verify the model can do a forward pass.
    """

    @pytest.fixture
    def source_weights(self, llava_pixtral_checkpoint):
        """Load source weights from the Llava checkpoint."""
        from safetensors.torch import load_file

        weight_files = list(llava_pixtral_checkpoint.glob("*.safetensors"))
        weights = {}
        for f in weight_files:
            weights.update(load_file(f))
        return weights

    @pytest.fixture
    def source_config(self, llava_pixtral_checkpoint):
        """Load source config from the Llava checkpoint."""
        with open(llava_pixtral_checkpoint / "config.json") as f:
            return json.load(f)

    def test_initial_conversion_produces_working_model(self, source_config, source_weights):
        """Test that Llava → Apriel2 conversion produces a working model."""
        # Convert config
        apriel2_config_dict = convert_llava_config(source_config)

        # Build and execute plan
        plan = plan_llava_to_apriel2(source_config)
        apriel2_weights = execute(plan, source_weights, seed=0)

        # Instantiate model
        config = Apriel2Config(**apriel2_config_dict)
        model = Apriel2ForConditionalGeneration(config)

        # Load weights (handle missing keys gracefully for vision encoder)
        model.load_state_dict(apriel2_weights, strict=False)

        # Forward pass
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.logits.shape == (1, 8, config.vocab_size)

    def test_each_surgery_step_produces_working_model(self, source_config, source_weights, additive_surgery_chain):
        """Test that each surgery step produces a model that can forward pass.

        Key insight: Surgery plans reference Apriel2 keys, so we must COMPOSE
        them with the conversion plan, not execute them on converted weights.
        The composed plan is then executed on the ORIGINAL source weights.
        """
        # Initial Llava → Apriel2 conversion
        apriel2_config = convert_llava_config(source_config)
        conversion_plan = plan_llava_to_apriel2(source_config)

        # Verify initial model works (conversion plan only)
        initial_weights = execute(conversion_plan, source_weights, seed=0)
        config = Apriel2Config(**apriel2_config)
        model = Apriel2ForConditionalGeneration(config)
        model.load_state_dict(initial_weights, strict=False)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)
        assert outputs.logits is not None, "Initial model forward pass failed"

        # Build cumulative plan: conversion | surgery_1 | surgery_2 | ...
        current_plan = conversion_plan
        current_config = apriel2_config

        for i, surgery in enumerate(additive_surgery_chain):
            # Compose config FIRST to get full target config (strips init)
            target_config = compose_configs(current_config, surgery)

            # Build plan from surgery spec (which has init fields)
            surgery_plan = plan_surgery(current_config, surgery)

            # Compose with current plan
            current_plan = compose(current_plan, surgery_plan)

            # Update current config
            current_config = target_config

            # Execute the composed plan on ORIGINAL source weights
            new_weights = execute(current_plan, source_weights, seed=0)

            # Verify config is valid
            try:
                config = Apriel2Config(**current_config)
            except Exception as e:
                pytest.fail(f"Step {i+1}: Invalid config - {e}")

            # Instantiate model
            try:
                model = Apriel2ForConditionalGeneration(config)
            except Exception as e:
                pytest.fail(f"Step {i+1}: Failed to instantiate model - {e}")

            # Load weights
            try:
                model.load_state_dict(new_weights, strict=False)
            except Exception as e:
                pytest.fail(f"Step {i+1}: Failed to load weights - {e}")

            # Forward pass
            input_ids = torch.randint(0, config.vocab_size, (1, 8))
            with torch.no_grad():
                try:
                    outputs = model(input_ids)
                    assert outputs.logits.shape == (1, 8, config.vocab_size)
                except Exception as e:
                    pytest.fail(f"Step {i+1}: Forward pass failed - {e}")

    def test_all_stochastic_submixers_via_cycling(self, source_config, source_weights, additive_surgery_chain):
        """Test ALL sub-mixers in stochastic blocks, not just the main mixer.

        Problem: Forward pass only exercises the main_mixer_name. Other sub-mixers
        could have bugs (wrong shapes, NaN weights, missing keys) and we'd never know.

        Solution: After each surgery that produces stochastic mixers, insert cycling
        surgeries that change main_mixer_name to test each sub-mixer, then restore.

        This validates:
        1. All sub-mixer weights are valid
        2. All sub-mixers can produce a forward pass
        3. Cycling surgeries (pure config changes) compose correctly
        4. Passthrough plans work correctly
        """
        # Initial Llava → Apriel2 conversion
        apriel2_config = convert_llava_config(source_config)
        conversion_plan = plan_llava_to_apriel2(source_config)

        # Expand surgery chain with cycling
        expanded_chain = expand_surgery_chain_with_cycling(additive_surgery_chain, apriel2_config)

        # Build cumulative plan: conversion | surgery_1 | cycling_1a | ... | restore_1 | surgery_2 | ...
        current_plan = conversion_plan
        current_config = apriel2_config

        for surgery, desc, is_restore in expanded_chain:
            # Compose config
            target_config = compose_configs(current_config, surgery)

            # Build and compose plan
            surgery_plan = plan_surgery(current_config, surgery)
            current_plan = compose(current_plan, surgery_plan)
            current_config = target_config

            # Execute the composed plan on ORIGINAL source weights
            new_weights = execute(current_plan, source_weights, seed=0)

            # Verify config is valid
            try:
                config = Apriel2Config(**current_config)
            except Exception as e:
                pytest.fail(f"{desc}: Invalid config - {e}")

            # Instantiate model
            try:
                model = Apriel2ForConditionalGeneration(config)
            except Exception as e:
                pytest.fail(f"{desc}: Failed to instantiate model - {e}")

            # Load weights
            try:
                model.load_state_dict(new_weights, strict=False)
            except Exception as e:
                pytest.fail(f"{desc}: Failed to load weights - {e}")

            # Forward pass (even for restore - validates state consistency)
            input_ids = torch.randint(0, config.vocab_size, (1, 8))
            with torch.no_grad():
                try:
                    outputs = model(input_ids)
                    assert outputs.logits.shape == (1, 8, config.vocab_size)
                except Exception as e:
                    pytest.fail(f"{desc}: Forward pass failed - {e}")

    def test_composed_plan_equals_sequential_execution(self, source_config, source_weights, additive_surgery_chain):
        """Test that composing plans gives same result as sequential execution.

        This verifies plan composition associativity:
        execute(compose(plan_A, plan_B), weights) == execute(plan_B, execute(plan_A, weights))
        """
        # Initial conversion
        base_config = convert_llava_config(source_config)
        conversion_plan = plan_llava_to_apriel2(source_config)
        base_weights = execute(conversion_plan, source_weights, seed=0)

        # Build all surgery plans
        plans = []
        configs = [base_config]
        config = base_config
        for surgery in additive_surgery_chain:
            # Compose config FIRST to get full target config
            target_config = compose_configs(config, surgery)
            # Build plan for this surgery (source→target, both complete configs)
            plan = plan_surgery(config, target_config)
            plans.append(plan)
            config = target_config
            configs.append(config)

        # Sequential execution
        seq_weights = base_weights
        for plan in plans:
            seq_weights = execute(plan, seq_weights, seed=0)

        # Composed execution
        composed_plan = plans[0]
        for plan in plans[1:]:
            composed_plan = compose(composed_plan, plan)
        composed_weights = execute(composed_plan, base_weights, seed=0)

        # Compare weights
        for key in seq_weights:
            if key in composed_weights:
                assert torch.allclose(seq_weights[key], composed_weights[key], atol=1e-5), f"Weight mismatch for {key}"

    def test_final_model_structure(self, source_config, source_weights, additive_surgery_chain):
        """Verify the final model has the expected structure."""
        # Initial conversion
        current_config = convert_llava_config(source_config)
        conversion_plan = plan_llava_to_apriel2(source_config)
        current_weights = execute(conversion_plan, source_weights, seed=0)

        # Apply all surgeries
        for i, surgery in enumerate(additive_surgery_chain):
            # Compose config for model instantiation (strips init)
            target_config = compose_configs(current_config, surgery)
            # Build plan from surgery spec (which has init fields)
            surgery_plan = plan_surgery(current_config, surgery)
            current_weights = execute(surgery_plan, current_weights, seed=i)
            current_config = target_config

        # Verify final structure
        mixer = current_config["decoder"]["block"]["mixer"]
        assert mixer["type"] == "stochastic"
        assert "attention" in mixer["mixers"]
        assert "sliding_window" in mixer["mixers"]
        assert "gdn" in mixer["mixers"]

        # Verify sub-mixers have correct types
        assert mixer["mixers"]["attention"]["type"] == "attention"
        assert mixer["mixers"]["sliding_window"]["type"] == "attention"
        assert mixer["mixers"]["sliding_window"]["window_size"] == 512
        assert mixer["mixers"]["gdn"]["type"] == "gdn"

        # Verify model works
        config = Apriel2Config(**current_config)
        model = Apriel2ForConditionalGeneration(config)
        model.load_state_dict(current_weights, strict=False)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)
        assert outputs.logits.shape == (1, 8, config.vocab_size)

    def test_plan_associativity(self, source_config, source_weights, additive_surgery_chain):
        """Test that plan composition is associative.

        compose(compose(A, B), C) == compose(A, compose(B, C))
        """
        # Initial conversion
        base_config = convert_llava_config(source_config)

        # Build surgery plans
        plans = []
        config = base_config
        for surgery in additive_surgery_chain:
            # Compose config FIRST to get full target config
            target_config = compose_configs(config, surgery)
            # Build plan for this surgery (source→target, both complete configs)
            plan = plan_surgery(config, target_config)
            plans.append(plan)
            config = target_config

        if len(plans) >= 3:
            A, B, C = plans[0], plans[1], plans[2]

            # Left-associated: (A | B) | C
            left = compose(compose(A, B), C)

            # Right-associated: A | (B | C)
            right = compose(A, compose(B, C))

            # Plans should be equivalent (same target expressions)
            assert set(left.mappings.keys()) == set(right.mappings.keys()), "Plan keys should match"

            # Execute both and compare results
            conversion_plan = plan_llava_to_apriel2(source_config)
            base_weights = execute(conversion_plan, source_weights, seed=0)

            left_weights = execute(left, base_weights, seed=0)
            right_weights = execute(right, base_weights, seed=0)

            for key in left_weights:
                if key in right_weights:
                    assert torch.allclose(
                        left_weights[key], right_weights[key], atol=1e-5
                    ), f"Associativity failed for {key}"


class TestPlanConfigConsistency:
    """Test that plan composition is consistent with config composition.

    Key property: For any way of grouping surgeries [S1, ..., Sn]:
    - Direct: plan_surgery(base, final_config)
    - Via groups: compose(plan_G1, plan_G2, ..., plan_Gm)

    These should produce identical weights when executed.
    """

    @pytest.fixture
    def base_setup(self, llava_pixtral_checkpoint):
        """Set up base config and weights after Llava conversion."""
        from safetensors.torch import load_file

        from fast_llm_external_models.apriel2.conversion.llava import convert_config as convert_llava_config

        # Load source config and weights
        with open(llava_pixtral_checkpoint / "config.json") as f:
            source_config = json.load(f)

        weight_files = list(llava_pixtral_checkpoint.glob("*.safetensors"))
        source_weights = {}
        for wf in weight_files:
            source_weights.update(load_file(wf))

        # Convert to Apriel2
        base_config = convert_llava_config(source_config)
        conversion_plan = plan_llava_to_apriel2(source_config)
        base_weights = execute(conversion_plan, source_weights, seed=0)
        return base_config, base_weights

    def _merge_surgeries(self, surgeries: list[dict]) -> dict:
        """Merge a list of surgery specs into one."""
        from fast_llm_external_models.apriel2.conversion.config import _deep_merge

        if not surgeries:
            return {}
        result = surgeries[0]
        for s in surgeries[1:]:
            result = _deep_merge(result, s)
        return result

    def _build_incremental_plans(self, base_config: dict, surgeries: list[dict]) -> tuple[list, list[dict]]:
        """Build incremental plans for each surgery step.

        Returns (plans, configs) where configs[i] is the config after surgery i.
        """
        plans = []
        configs = [base_config]
        config = base_config
        for surgery in surgeries:
            target_config = compose_configs(config, surgery)
            plan = plan_surgery(config, target_config)
            plans.append(plan)
            configs.append(target_config)
            config = target_config
        return plans, configs

    def test_incremental_equals_direct_full_chain(self, base_setup, additive_surgery_chain):
        """Test that composing all incremental plans equals one direct plan.

        compose(P1, P2, ..., Pn) ≡ plan_surgery(base, final)
        """
        base_config, base_weights = base_setup
        surgeries = additive_surgery_chain

        # Build incremental plans
        plans, configs = self._build_incremental_plans(base_config, surgeries)
        final_config = configs[-1]

        # Compose all incremental plans
        composed_plan = plans[0]
        for plan in plans[1:]:
            composed_plan = compose(composed_plan, plan)

        # Build direct plan
        direct_plan = plan_surgery(base_config, final_config)

        # Verify same target keys
        assert set(composed_plan.mappings.keys()) == set(direct_plan.mappings.keys()), "Plan keys should match"

        # Execute both and compare weights
        composed_weights = execute(composed_plan, base_weights, seed=0)
        direct_weights = execute(direct_plan, base_weights, seed=0)

        for key in direct_weights:
            assert torch.allclose(
                composed_weights[key], direct_weights[key], atol=1e-5
            ), f"Incremental vs direct mismatch for {key}"

    def test_every_prefix_consistency(self, base_setup, additive_surgery_chain):
        """Test that every prefix of the surgery chain satisfies consistency.

        For k = 1, 2, ..., n:
        compose(P1, ..., Pk) ≡ plan_surgery(base, config_k)
        """
        base_config, base_weights = base_setup
        surgeries = additive_surgery_chain

        # Build all incremental plans
        plans, configs = self._build_incremental_plans(base_config, surgeries)

        # Test each prefix
        for k in range(1, len(surgeries) + 1):
            # Compose first k plans
            composed = plans[0]
            for plan in plans[1:k]:
                composed = compose(composed, plan)

            # Direct plan to config_k
            direct = plan_surgery(base_config, configs[k])

            # Verify keys match
            assert set(composed.mappings.keys()) == set(direct.mappings.keys()), f"Prefix {k}: keys don't match"

            # Execute and compare
            composed_weights = execute(composed, base_weights, seed=0)
            direct_weights = execute(direct, base_weights, seed=0)

            for key in direct_weights:
                assert torch.allclose(
                    composed_weights[key], direct_weights[key], atol=1e-5
                ), f"Prefix {k} mismatch for {key}"

    def test_every_binary_split_consistency(self, base_setup, additive_surgery_chain):
        """Test every binary split of the surgery chain.

        For each split point k:
        - G1 = merge(S1, ..., Sk)
        - G2 = merge(Sk+1, ..., Sn)
        - compose(plan_G1, plan_G2) ≡ plan_surgery(base, final)
        """
        base_config, base_weights = base_setup
        surgeries = additive_surgery_chain
        n = len(surgeries)

        if n < 2:
            pytest.skip("Need at least 2 surgeries for binary split test")

        # Build direct plan to final config
        merged_all = self._merge_surgeries(surgeries)
        final_config = compose_configs(base_config, merged_all)
        direct_plan = plan_surgery(base_config, final_config)
        direct_weights = execute(direct_plan, base_weights, seed=0)

        # Test each binary split
        for split_point in range(1, n):
            # Group 1: surgeries [0, split_point)
            merged_g1 = self._merge_surgeries(surgeries[:split_point])
            config_g1 = compose_configs(base_config, merged_g1)
            plan_g1 = plan_surgery(base_config, config_g1)

            # Group 2: surgeries [split_point, n)
            merged_g2 = self._merge_surgeries(surgeries[split_point:])
            config_g2 = compose_configs(config_g1, merged_g2)
            plan_g2 = plan_surgery(config_g1, config_g2)

            # Compose the two group plans
            split_plan = compose(plan_g1, plan_g2)

            # Verify final configs are equal (sanity check)
            assert config_g2 == final_config, f"Split {split_point}: configs don't match"

            # Verify keys match
            assert set(split_plan.mappings.keys()) == set(
                direct_plan.mappings.keys()
            ), f"Split {split_point}: keys don't match"

            # Execute and compare
            split_weights = execute(split_plan, base_weights, seed=0)

            for key in direct_weights:
                assert torch.allclose(
                    split_weights[key], direct_weights[key], atol=1e-5
                ), f"Binary split at {split_point} failed for {key}"

    def test_all_partitions_consistency(self, base_setup, additive_surgery_chain):
        """Test that ALL partitions of the surgery chain give the same result.

        For a chain [A, B, C], test partitions like:
        - [[A], [B], [C]] (fully incremental)
        - [[A, B], [C]] (merge first two)
        - [[A], [B, C]] (merge last two)
        - [[A, B, C]] (fully merged / direct)

        All should produce identical weights.
        """
        from itertools import combinations

        base_config, base_weights = base_setup
        surgeries = additive_surgery_chain
        n = len(surgeries)

        if n < 2:
            pytest.skip("Need at least 2 surgeries for partition test")

        # Reference: direct plan
        merged_all = self._merge_surgeries(surgeries)
        final_config = compose_configs(base_config, merged_all)
        direct_plan = plan_surgery(base_config, final_config)
        reference_weights = execute(direct_plan, base_weights, seed=0)

        def generate_partitions(n: int):
            """Generate all ways to partition [0, 1, ..., n-1] into contiguous groups."""
            if n == 0:
                yield []
                return
            if n == 1:
                yield [[0]]
                return

            # Split points between elements (n-1 possible split points)
            # Each subset of split points gives a partition
            for num_splits in range(n):  # 0 to n-1 splits
                for split_points in combinations(range(1, n), num_splits):
                    # Convert split points to partition
                    partition = []
                    prev = 0
                    for sp in split_points:
                        partition.append(list(range(prev, sp)))
                        prev = sp
                    partition.append(list(range(prev, n)))
                    yield partition

        # Test all partitions
        partitions_tested = 0
        for partition in generate_partitions(n):
            # Build plan for this partition
            config = base_config
            plans = []

            for group_indices in partition:
                # Merge surgeries in this group
                group_surgeries = [surgeries[i] for i in group_indices]
                merged = self._merge_surgeries(group_surgeries)

                # Build plan for this group
                target_config = compose_configs(config, merged)
                plan = plan_surgery(config, target_config)
                plans.append(plan)
                config = target_config

            # Compose all group plans
            composed = plans[0]
            for plan in plans[1:]:
                composed = compose(composed, plan)

            # Execute and compare to reference
            partition_weights = execute(composed, base_weights, seed=0)

            partition_str = str([[surgeries[i] for i in g] for g in partition])[:100]
            for key in reference_weights:
                assert torch.allclose(
                    partition_weights[key], reference_weights[key], atol=1e-5
                ), f"Partition {partition} failed for {key}"

            partitions_tested += 1

        # Verify we tested a reasonable number of partitions
        # For n items, there are 2^(n-1) partitions
        expected = 2 ** (n - 1)
        assert partitions_tested == expected, f"Expected {expected} partitions, got {partitions_tested}"


class TestComprehensiveTortureChain:
    """Test the comprehensive torture chain with pattern decoders.

    This is the REAL stress test exercising:
    - Fixed → Pattern decoder transitions
    - Per-layer heterogeneity (different mixers per layer)
    - All type conversions: FA ↔ SWA ↔ Mamba ↔ GDN
    - Stochastic wrapping/unwrapping
    - Both init: transfer and init: random
    - Destructive operations
    """

    @pytest.fixture
    def torture_setup(self, llava_pixtral_checkpoint):
        """Set up for comprehensive torture tests."""
        from safetensors.torch import load_file

        from fast_llm_external_models.apriel2.conversion.llava import convert_config as convert_llava_config

        # Load source
        with open(llava_pixtral_checkpoint / "config.json") as f:
            source_config = json.load(f)

        weight_files = list(llava_pixtral_checkpoint.glob("*.safetensors"))
        source_weights = {}
        for wf in weight_files:
            source_weights.update(load_file(wf))

        # Convert to Apriel2
        base_config = convert_llava_config(source_config)
        conversion_plan = plan_llava_to_apriel2(source_config)
        base_weights = execute(conversion_plan, source_weights, seed=0)

        return base_config, base_weights

    def test_each_step_produces_valid_config(self, torture_setup, comprehensive_torture_chain):
        """Test that each surgery step produces a valid config."""
        base_config, _ = torture_setup

        current_config = base_config
        for i, surgery in enumerate(comprehensive_torture_chain):
            try:
                current_config = compose_configs(current_config, surgery)
                # Verify it's a valid Apriel2Config
                config = Apriel2Config(**current_config)
                assert config is not None
            except Exception as e:
                pytest.fail(f"Step {i+1} produced invalid config: {e}")

    @requires_cuda
    def test_each_step_produces_working_model(self, torture_setup, comprehensive_torture_chain):
        """Test that each surgery step produces a model that can forward pass.

        This is the ultimate integration test - config composition + plan building
        + weight conversion + model instantiation + forward pass.
        """
        base_config, base_weights = torture_setup

        current_config = base_config
        current_weights = base_weights

        for i, surgery in enumerate(comprehensive_torture_chain):
            # Compose config (strips init, used for model instantiation)
            target_config = compose_configs(current_config, surgery)

            # Build plan from surgery spec (which has init fields)
            # Note: plan_surgery needs the surgery spec with init fields,
            # not the composed config (which has init stripped)
            try:
                surgery_plan = plan_surgery(current_config, surgery)
            except Exception as e:
                pytest.fail(f"Step {i+1}: plan_surgery failed - {e}")

            # Execute plan
            try:
                new_weights = execute(surgery_plan, current_weights, seed=i)
            except Exception as e:
                pytest.fail(f"Step {i+1}: execute failed - {e}")

            # Instantiate model
            try:
                config = Apriel2Config(**target_config)
                model = Apriel2ForConditionalGeneration(config)
            except Exception as e:
                pytest.fail(f"Step {i+1}: model instantiation failed - {e}")

            # Load weights
            try:
                model.load_state_dict(new_weights, strict=False)
            except Exception as e:
                pytest.fail(f"Step {i+1}: load_state_dict failed - {e}")

            # Forward pass
            try:
                input_ids = torch.randint(0, config.vocab_size, (1, 8))
                with torch.no_grad():
                    outputs = model(input_ids)
                assert outputs.logits.shape == (1, 8, config.vocab_size)
            except Exception as e:
                pytest.fail(f"Step {i+1}: forward pass failed - {e}")

            current_config = target_config
            current_weights = new_weights

    @requires_cuda
    def test_final_supernet_structure(self, torture_setup, comprehensive_torture_chain):
        """Verify the final architecture has supernet blocks with all 4 mixer types."""
        base_config, base_weights = torture_setup

        # Apply all surgeries
        current_config = base_config
        current_weights = base_weights
        for i, surgery in enumerate(comprehensive_torture_chain):
            target_config = compose_configs(current_config, surgery)
            plan = plan_surgery(current_config, surgery)  # Use surgery spec (has init)
            current_weights = execute(plan, current_weights, seed=i)
            current_config = target_config

        # Verify final structure - pattern decoder with heterogeneous blocks
        assert current_config["decoder"]["type"] == "pattern"
        blocks = current_config["decoder"]["blocks"]

        # Verify supernet block has all 4 mixer types
        assert "supernet" in blocks, "Should have supernet block"
        supernet_mixer = blocks["supernet"]["mixer"]
        assert supernet_mixer["type"] == "stochastic"
        assert "attention" in supernet_mixer["mixers"]
        assert "swa" in supernet_mixer["mixers"]
        assert "mamba" in supernet_mixer["mixers"]
        assert "gdn" in supernet_mixer["mixers"]

        # Verify model works
        config = Apriel2Config(**current_config)
        model = Apriel2ForConditionalGeneration(config)
        model.load_state_dict(current_weights, strict=False)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)
        assert outputs.logits.shape == (1, 8, config.vocab_size)

    @requires_cuda
    def test_plan_config_consistency_comprehensive(self, torture_setup, comprehensive_torture_chain):
        """Test that incremental plan composition works for the comprehensive chain.

        Note: We cannot compare to a "direct plan" because the comprehensive chain
        has intermediate `init: random` steps. A direct plan from base to final
        would not know which parts need random init, so it would give different
        results than the composed incremental plans.

        Instead, we verify that:
        1. Each incremental plan builds successfully using surgery specs (with init)
        2. Plans can be composed together
        3. The composed plan executes successfully
        """
        base_config, base_weights = torture_setup
        surgeries = comprehensive_torture_chain

        # Build incremental plans using surgery specs (which have init fields)
        plans = []
        config = base_config
        for surgery in surgeries:
            # Use surgery spec (has init), not composed config (no init)
            plan = plan_surgery(config, surgery)
            plans.append(plan)
            # Update config for next iteration
            config = compose_configs(config, surgery)
        final_config = config

        # Compose all incremental plans
        composed_plan = plans[0]
        for plan in plans[1:]:
            composed_plan = compose(composed_plan, plan)

        # Execute the composed plan
        final_weights = execute(composed_plan, base_weights, seed=0)

        # Verify model instantiation works with final config and weights
        model_config = Apriel2Config(**final_config)
        model = Apriel2ForConditionalGeneration(model_config)
        model.load_state_dict(final_weights, strict=False)

        # Verify forward pass works
        input_ids = torch.randint(0, model_config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)
        assert outputs.logits.shape == (1, 8, model_config.vocab_size)


class TestPlanCompositionWithRealYAML:
    """Test plan composition using real YAML surgery files."""

    @requires_cuda
    def test_stochastic_supernet_yaml_end_to_end(self, llava_pixtral_checkpoint):
        """Test full pipeline with stochastic_supernet.yaml."""
        import yaml
        from safetensors.torch import load_file

        # Load source
        with open(llava_pixtral_checkpoint / "config.json") as f:
            source_config = json.load(f)

        weight_files = list(llava_pixtral_checkpoint.glob("*.safetensors"))
        source_weights = {}
        for f in weight_files:
            source_weights.update(load_file(f))

        # Load surgery YAML
        yaml_path = Path(__file__).parent.parent.parent / "apriel2" / "examples" / "stochastic_supernet.yaml"
        with open(yaml_path) as f:
            surgery_config = yaml.safe_load(f)

        # Convert config
        apriel2_config = convert_llava_config(source_config)

        # Build full plan: Llava → Apriel2 → Surgery
        conversion_plan = plan_llava_to_apriel2(source_config)
        surgery_plan = plan_surgery(apriel2_config, surgery_config)
        full_plan = compose(conversion_plan, surgery_plan)

        # Execute
        final_weights = execute(full_plan, source_weights, seed=0)

        # Compose config
        final_config = compose_configs(apriel2_config, surgery_config)

        # Verify model works
        config = Apriel2Config(**final_config)
        model = Apriel2ForConditionalGeneration(config)
        model.load_state_dict(final_weights, strict=False)

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids)

        assert outputs.logits.shape == (1, 8, config.vocab_size)

        # Verify stochastic mixer structure
        mixer = config.decoder["block"]["mixer"]
        assert mixer["type"] == "stochastic"
        assert "attention" in mixer["mixers"]
        assert "sliding_window" in mixer["mixers"]
        assert "gdn" in mixer["mixers"]


class TestInitSeparationOfConcerns:
    """Tests verifying that init mode is ONLY about weights, not config structure.

    Key principles:
    1. Config composition should produce identical structure regardless of init mode
    2. plan_surgery with init: random should succeed for ANY type pair
    3. plan_surgery with init: transfer should fail for unsupported type pairs
    4. The init field is metadata for the plan builder, not the config composer
    """

    @pytest.fixture
    def base_config(self):
        """Simple base config with attention mixer."""
        return {
            "model_type": "apriel2",
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": {
                        "type": "attention",
                        "heads": 8,
                        "head_groups": 4,
                        "head_size": 32,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

    @pytest.fixture
    def mamba_config(self):
        """Config with mamba mixer."""
        return {
            "model_type": "apriel2",
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": {
                        "type": "mamba",
                        "d_inner": 256,
                        "d_xb": 64,
                        "dt_rank": 16,
                        "d_state": 16,
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

    def test_config_composition_identical_regardless_of_init_mode(self, base_config):
        """Config composition produces same structure with init: transfer vs init: random."""
        # Surgery with init: transfer
        surgery_transfer = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "init": "transfer"},
                            "swa": {
                                "type": "attention",
                                "init": "transfer",
                                "sliding_window": 512,
                            },
                        },
                    },
                },
            },
        }

        # Surgery with init: random
        surgery_random = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "init": "random"},
                            "swa": {
                                "type": "attention",
                                "init": "random",
                                "sliding_window": 512,
                            },
                        },
                    },
                },
            },
        }

        # Compose configs
        result_transfer = compose_configs(base_config, surgery_transfer)
        result_random = compose_configs(base_config, surgery_random)

        # Both should produce identical structure (init is stripped)
        assert (
            result_transfer == result_random
        ), "Config composition should produce identical structure regardless of init mode"

        # Verify the structure is correct
        mixer = result_transfer["decoder"]["block"]["mixer"]
        assert mixer["type"] == "stochastic"
        assert "attention" in mixer["mixers"]
        assert "swa" in mixer["mixers"]
        # init should be stripped
        assert "init" not in mixer["mixers"]["attention"]
        assert "init" not in mixer["mixers"]["swa"]

    def test_plan_surgery_random_succeeds_for_any_type_pair(self, mamba_config):
        """plan_surgery with init: random should succeed even for mamba -> attention."""
        # This surgery changes mamba to attention with random init
        # There's no mamba->attention converter, but init: random doesn't need one
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "attention",
                        "init": "random",
                        "heads": 8,
                        "head_groups": 4,
                        "head_size": 32,
                        "rotary": {"type": "mistral_1d", "theta": 10000.0},
                    },
                },
            },
        }

        # This should NOT raise - init: random doesn't need a converter
        plan = plan_surgery(mamba_config, surgery)

        # Verify the plan has the expected target keys
        target_keys = {str(k) for k in plan.mappings.keys()}
        assert any("mixer.q_proj" in k for k in target_keys)

    def test_plan_surgery_transfer_fails_for_unsupported_type_pair(self, mamba_config):
        """plan_surgery with init: transfer should fail for mamba -> attention."""
        # This surgery changes mamba to attention with transfer init
        # There's no mamba->attention converter, so this should fail
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "attention",
                        "init": "transfer",
                        "heads": 8,
                        "head_groups": 4,
                        "head_size": 32,
                    },
                },
            },
        }

        # This should raise because there's no mamba->attention converter
        with pytest.raises(ValueError, match="No converter available for mamba -> attention"):
            plan_surgery(mamba_config, surgery)

    def test_plan_surgery_transfer_succeeds_for_supported_type_pair(self, base_config):
        """plan_surgery with init: transfer succeeds for attention -> mamba (MIL)."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "mamba",
                        "init": "transfer",
                        "d_inner": 256,
                        "d_xb": 64,
                        "dt_rank": 16,
                        "d_state": 16,
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
        }

        # This should succeed - attention->mamba has MIL converter
        plan = plan_surgery(base_config, surgery)

        # Verify the plan has mamba target keys
        target_keys = {str(k) for k in plan.mappings.keys()}
        assert any("mixer.in_proj" in k for k in target_keys)

    def test_stochastic_init_random_succeeds_for_any_submixer_type(self, mamba_config):
        """Stochastic mixer with init: random sub-mixers succeeds regardless of source."""
        # Source is mamba, target is stochastic with attention sub-mixers
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {
                                "type": "attention",
                                "init": "random",
                                "heads": 8,
                                "head_groups": 4,
                                "head_size": 32,
                                "rotary": {"type": "mistral_1d", "theta": 10000.0},
                            },
                            "swa": {
                                "type": "attention",
                                "init": "random",
                                "heads": 8,
                                "head_groups": 4,
                                "head_size": 32,
                                "sliding_window": 512,
                                "rotary": {"type": "mistral_1d", "theta": 10000.0},
                            },
                        },
                    },
                },
            },
        }

        # This should succeed - init: random doesn't need converters
        plan = plan_surgery(mamba_config, surgery)

        # Verify both sub-mixers have target keys
        target_keys = {str(k) for k in plan.mappings.keys()}
        assert any("mixers.attention.q_proj" in k for k in target_keys)
        assert any("mixers.swa.q_proj" in k for k in target_keys)

    def test_mixed_init_modes_in_stochastic(self, base_config):
        """Stochastic mixer can have some sub-mixers transfer, others random."""
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            # This can transfer from source attention
                            "attention": {"type": "attention", "init": "transfer"},
                            # This must be random (no gdn->attention transfer on source)
                            "gdn": {
                                "type": "gdn",
                                "init": "random",
                                "value_heads": 8,
                                "key_heads": 4,
                                "key_head_dim": 32,
                                "value_head_dim": 32,
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                },
            },
        }

        # This should succeed
        plan = plan_surgery(base_config, surgery)

        # Verify both sub-mixers have target keys
        target_keys = {str(k) for k in plan.mappings.keys()}
        assert any("mixers.attention.q_proj" in k for k in target_keys)
        assert any("mixers.gdn.in_proj_qkvz" in k for k in target_keys)


class TestMarkovianProperty:
    """Tests verifying that plan creation is Markovian.

    The Markovian property states: plan_surgery(current_config, surgery)
    depends ONLY on current_config and surgery, NOT on the history of
    how we arrived at current_config.

    This is essential for associativity of composition:
        compose(compose(A, B), C) == compose(A, compose(B, C))

    If plans depended on history, associativity would break.
    """

    @pytest.fixture
    def attention_config(self):
        """Base config with attention."""
        return {
            "model_type": "apriel2",
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": {
                        "type": "attention",
                        "heads": 8,
                        "head_groups": 4,
                        "head_size": 32,
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

    @pytest.fixture
    def stochastic_config(self):
        """Config with stochastic mixer."""
        return {
            "model_type": "apriel2",
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {
                                "type": "attention",
                                "heads": 8,
                                "head_groups": 4,
                                "head_size": 32,
                            },
                            "swa": {
                                "type": "sliding_window",
                                "heads": 8,
                                "head_groups": 4,
                                "head_size": 32,
                                "window_size": 512,
                            },
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

    def test_different_paths_same_config_same_plan(self, attention_config):
        """Two different paths to the same config produce identical plans.

        Path A: attention -> stochastic{att, swa}
        Path B: attention -> stochastic{att} -> stochastic{att, swa}

        If the final configs are identical, the plans must be identical.
        """
        # Path A: Direct to stochastic with both sub-mixers
        surgery_a = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "init": "transfer"},
                            "swa": {
                                "type": "sliding_window",
                                "init": "transfer",
                                "window_size": 512,
                            },
                        },
                    },
                },
            },
        }
        config_a = compose_configs(attention_config, surgery_a)

        # Path B: First add attention only, then add swa
        surgery_b1 = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "init": "transfer"},
                        },
                    },
                },
            },
        }
        intermediate_config = compose_configs(attention_config, surgery_b1)

        surgery_b2 = {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "swa": {
                                "type": "sliding_window",
                                "init": "transfer",
                                "window_size": 512,
                            },
                        },
                    },
                },
            },
        }
        config_b = compose_configs(intermediate_config, surgery_b2)

        # The configs should be identical (both have att and swa)
        assert config_a == config_b, "Different paths should produce same config"

        # Now apply the SAME surgery to both configs
        final_surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "gdn": {
                                "type": "gdn",
                                "init": "transfer",
                                "value_heads": 8,
                                "key_heads": 4,
                                "key_head_dim": 32,
                                "value_head_dim": 32,
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                },
            },
        }

        # Plans should be identical because:
        # 1. Source configs (config_a, config_b) are identical
        # 2. Surgery is identical
        # 3. Plan depends only on source and surgery (Markovian)
        plan_from_a = plan_surgery(config_a, final_surgery)
        plan_from_b = plan_surgery(config_b, final_surgery)

        # Compare plan mappings
        keys_a = {str(k) for k in plan_from_a.mappings.keys()}
        keys_b = {str(k) for k in plan_from_b.mappings.keys()}
        assert keys_a == keys_b, "Plans from same config via different paths should be identical"

    def test_init_in_source_config_does_not_affect_plan(self, attention_config):
        """Manually injecting init into source config doesn't change the plan.

        This tests that plan_surgery reads init from surgery, not source.
        (Note: This is an artificial test - compose_configs strips init,
        so in practice source configs never have init fields.)
        """
        import copy

        # Create two copies of the config
        config_with_init = copy.deepcopy(attention_config)
        config_without_init = copy.deepcopy(attention_config)

        # Manually inject init into one (bypassing compose_configs)
        config_with_init["decoder"]["block"]["mixer"]["init"] = "random"

        # Same surgery
        surgery = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "init": "transfer"},
                        },
                    },
                },
            },
        }

        # Plans should depend on surgery's init, not source's init
        plan_with = plan_surgery(config_with_init, surgery)
        plan_without = plan_surgery(config_without_init, surgery)

        keys_with = {str(k) for k in plan_with.mappings.keys()}
        keys_without = {str(k) for k in plan_without.mappings.keys()}

        # Plans should be identical - source's init field is ignored
        assert keys_with == keys_without, "Plan should not depend on init in source config"

    def test_associativity_of_surgery_composition(self, attention_config):
        """Verify associativity: (A ∘ B) ∘ C == A ∘ (B ∘ C) for surgery specs.

        This tests that composing surgeries is associative, which is
        equivalent to Markovianity for plan creation.
        """
        surgery_a = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "init": "transfer"},
                        },
                    },
                },
            },
        }

        surgery_b = {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "swa": {
                                "type": "sliding_window",
                                "init": "transfer",
                                "window_size": 512,
                            },
                        },
                    },
                },
            },
        }

        surgery_c = {
            "decoder": {
                "block": {
                    "mixer": {
                        "mixers": {
                            "gdn": {
                                "type": "gdn",
                                "init": "random",
                                "value_heads": 8,
                                "key_heads": 4,
                                "key_head_dim": 32,
                                "value_head_dim": 32,
                                "convolution_layer": {"kernel_size": 4},
                            },
                        },
                    },
                },
            },
        }

        # Left association: ((attention_config ∘ A) ∘ B) ∘ C
        left_1 = compose_configs(attention_config, surgery_a)
        left_2 = compose_configs(left_1, surgery_b)
        left_result = compose_configs(left_2, surgery_c)

        # Right association: (attention_config ∘ A) ∘ (B ∘ C)
        # Note: B ∘ C is partial ∘ partial = deep merge of surgery specs
        bc_merged = compose_configs(surgery_b, surgery_c)
        right_1 = compose_configs(attention_config, surgery_a)
        right_result = compose_configs(right_1, bc_merged)

        assert left_result == right_result, "Surgery composition should be associative"

    def test_complete_configs_have_no_init_fields(self, attention_config):
        """Verify that compose_configs strips init from complete configs.

        This is the key invariant that enables Markovianity:
        - Complete configs (states) have no init fields
        - Surgery specs (transitions) have init fields
        - Plans read init from surgery, not state
        """
        surgery_with_init = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "init": "transfer"},
                            "swa": {"type": "sliding_window", "init": "random", "window_size": 512},
                        },
                    },
                },
            },
        }

        result = compose_configs(attention_config, surgery_with_init)

        # Recursively check for init fields
        def has_init(obj):
            if isinstance(obj, dict):
                if "init" in obj:
                    return True
                return any(has_init(v) for v in obj.values())
            if isinstance(obj, list):
                return any(has_init(v) for v in obj)
            return False

        assert not has_init(result), "Complete configs should have no init fields"

    def test_monoid_action_law_additive_surgeries(self):
        """Monoid action law HOLDS for additive surgeries.

        Additive surgeries (no type: declaration) support:
            apply(apply(s, t1), t2) == apply(s, t1 ∘ t2)

        This is because additive operations commute nicely:
        "add {a}" then "add {b}" == "add {a, b}"
        """
        # Start with stochastic (additive surgery target)
        s = {
            "model_type": "apriel2",
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {
                            "attention": {"type": "attention", "heads": 8, "head_groups": 4, "head_size": 32},
                        },
                    },
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

        # Additive surgeries (no type: declaration)
        t1 = {"decoder": {"block": {"mixer": {"mixers": {"swa": {"type": "sliding_window", "window_size": 512}}}}}}
        t2 = {"decoder": {"block": {"mixer": {"mixers": {"mamba": {"type": "mamba", "d_inner": 512}}}}}}

        # Path A: Sequential
        s_prime = compose_configs(s, t1)
        s_double_prime_A = compose_configs(s_prime, t2)

        # Path B: Composed
        t1_t2 = compose_configs(t1, t2)
        s_double_prime_B = compose_configs(s, t1_t2)

        assert s_double_prime_A == s_double_prime_B, "Monoid action law should hold for additive surgeries"

    def test_monoid_action_law_replacement_surgeries_fails(self):
        """Monoid action law FAILS for replacement surgeries (by design).

        Replacement surgeries (type: stochastic declared) have:
            apply(apply(s, t1), t2) != apply(s, t1 ∘ t2)

        This is FUNDAMENTAL, not a bug:
        - Sequential: "set to {a}" then "set to {b}" → {b} (second wins)
        - Composed: merge({a}, {b}) = {a,b}, then apply → {a,b}

        These are genuinely different semantics. The failure documents
        the distinction between declarative composition (merge) and
        operational composition (function application).
        """
        s = {
            "model_type": "apriel2",
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": {"type": "attention", "heads": 8, "head_groups": 4, "head_size": 32},
                    "mlp": {"type": "mlp", "intermediate_size": 256},
                    "normalization": {"type": "rms_norm", "epsilon": 1e-5},
                },
            },
        }

        # Replacement surgeries (both declare type: stochastic)
        t1 = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {"attention": {"type": "attention"}},
                    }
                }
            }
        }
        t2 = {
            "decoder": {
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "swa",
                        "mixers": {"swa": {"type": "sliding_window", "window_size": 512}},
                    }
                }
            }
        }

        # Path A: Sequential (second replacement wins)
        s_prime = compose_configs(s, t1)
        s_double_prime_A = compose_configs(s_prime, t2)

        # Path B: Composed (declarations merged)
        t1_t2 = compose_configs(t1, t2)
        s_double_prime_B = compose_configs(s, t1_t2)

        # They should be DIFFERENT (law fails)
        assert s_double_prime_A != s_double_prime_B, "Monoid action law should FAIL for replacement surgeries"

        # Verify the specific difference:
        # Sequential: only swa (second replacement wins)
        # Composed: both attention and swa (merged declarations)
        mixers_A = set(s_double_prime_A["decoder"]["block"]["mixer"]["mixers"].keys())
        mixers_B = set(s_double_prime_B["decoder"]["block"]["mixer"]["mixers"].keys())

        assert mixers_A == {"swa"}, "Sequential: second replacement wins"
        assert mixers_B == {"attention", "swa"}, "Composed: declarations merged"


class TestCyclingSurgeryGeneration:
    """Tests for the cycling surgery generation functions.

    These functions expand a surgery chain to test ALL sub-mixers in stochastic
    blocks, not just the main mixer.
    """

    def test_get_stochastic_blocks_fixed_decoder(self):
        """Test extraction of stochastic blocks from fixed decoder."""
        config = {
            "decoder": {
                "type": "fixed",
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {"attention": {}, "mamba": {}},
                    }
                },
            }
        }

        blocks = get_stochastic_blocks(config)

        assert "block" in blocks
        assert blocks["block"]["type"] == "stochastic"
        assert set(blocks["block"]["mixers"].keys()) == {"attention", "mamba"}

    def test_get_stochastic_blocks_pattern_decoder(self):
        """Test extraction of stochastic blocks from pattern decoder."""
        config = {
            "decoder": {
                "type": "pattern",
                "blocks": {
                    "attn": {"mixer": {"type": "attention"}},  # Not stochastic
                    "stoch": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "a",
                            "mixers": {"a": {}, "b": {}},
                        }
                    },
                },
            }
        }

        blocks = get_stochastic_blocks(config)

        assert len(blocks) == 1
        assert "blocks.stoch" in blocks
        assert "blocks.attn" not in blocks

    def test_get_stochastic_blocks_no_stochastic(self):
        """Test with config that has no stochastic blocks."""
        config = {
            "decoder": {
                "type": "fixed",
                "block": {"mixer": {"type": "attention"}},
            }
        }

        blocks = get_stochastic_blocks(config)

        assert blocks == {}

    def test_generate_cycling_surgeries_single_block(self):
        """Test cycling surgery generation for single stochastic block."""
        config = {
            "decoder": {
                "type": "fixed",
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {"attention": {}, "mamba": {}, "gdn": {}},
                    }
                },
            }
        }

        surgeries = generate_cycling_surgeries(config)

        # Should generate: cycle to mamba, cycle to gdn, restore to attention
        assert len(surgeries) == 3

        # Check cycling surgeries
        descs = [desc for _, desc in surgeries]
        assert "cycle block to mamba" in descs
        assert "cycle block to gdn" in descs
        assert "restore block to attention" in descs

        # Check surgery structure
        for surgery, desc in surgeries:
            assert "decoder" in surgery
            assert "block" in surgery["decoder"]
            assert "mixer" in surgery["decoder"]["block"]
            assert "main_mixer_name" in surgery["decoder"]["block"]["mixer"]

    def test_generate_cycling_surgeries_pattern_decoder(self):
        """Test cycling surgery generation for pattern decoder."""
        config = {
            "decoder": {
                "type": "pattern",
                "blocks": {
                    "a": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "x",
                            "mixers": {"x": {}, "y": {}},
                        }
                    },
                    "b": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "p",
                            "mixers": {"p": {}, "q": {}},
                        }
                    },
                },
            }
        }

        surgeries = generate_cycling_surgeries(config)

        # Block a: cycle to y, restore to x
        # Block b: cycle to q, restore to p
        assert len(surgeries) == 4

        descs = [desc for _, desc in surgeries]
        assert "cycle blocks.a to y" in descs
        assert "restore blocks.a to x" in descs
        assert "cycle blocks.b to q" in descs
        assert "restore blocks.b to p" in descs

    def test_generate_cycling_surgeries_single_submixer_no_cycling(self):
        """Test that single sub-mixer stochastic blocks don't generate cycling."""
        config = {
            "decoder": {
                "type": "fixed",
                "block": {
                    "mixer": {
                        "type": "stochastic",
                        "main_mixer_name": "attention",
                        "mixers": {"attention": {}},  # Only one sub-mixer
                    }
                },
            }
        }

        surgeries = generate_cycling_surgeries(config)

        # No cycling needed - only one sub-mixer
        assert surgeries == []

    def test_expand_surgery_chain_adds_cycling(self):
        """Test that expand_surgery_chain_with_cycling adds cycling surgeries."""
        initial_config = {
            "model_type": "apriel2",
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": {"type": "attention"},
                    "mlp": {},
                    "normalization": {},
                },
            },
        }

        surgery_chain = [
            # Convert to stochastic with two sub-mixers
            {
                "decoder": {
                    "block": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {"attention": {}, "mamba": {}},
                        }
                    }
                }
            }
        ]

        expanded = expand_surgery_chain_with_cycling(surgery_chain, initial_config)

        # Original surgery + cycle to mamba + restore to attention
        assert len(expanded) == 3

        descriptions = [desc for _, desc, _ in expanded]
        assert descriptions[0] == "surgery 1"
        assert descriptions[1] == "cycle block to mamba"
        assert descriptions[2] == "restore block to attention"

        # Verify restore flag
        assert expanded[0][2] is False  # surgery - not restore
        assert expanded[1][2] is False  # cycle - not restore
        assert expanded[2][2] is True  # restore

    def test_expand_surgery_chain_preserves_invariant(self):
        """Test that cycling leaves the chain state invariant."""
        initial_config = {
            "model_type": "apriel2",
            "hidden_size": 256,
            "vocab_size": 1000,
            "decoder": {
                "type": "fixed",
                "num_blocks": 2,
                "block": {
                    "mixer": {"type": "attention"},
                    "mlp": {},
                    "normalization": {},
                },
            },
        }

        surgery_chain = [
            {
                "decoder": {
                    "block": {
                        "mixer": {
                            "type": "stochastic",
                            "main_mixer_name": "attention",
                            "mixers": {"attention": {}, "mamba": {}},
                        }
                    }
                }
            }
        ]

        expanded = expand_surgery_chain_with_cycling(surgery_chain, initial_config)

        # Apply all surgeries and verify final state matches state after original surgery
        config_after_original = compose_configs(initial_config, surgery_chain[0])

        current_config = initial_config
        for surgery, desc, _ in expanded:
            current_config = compose_configs(current_config, surgery)

        # After cycling and restore, we should be back to the same state
        assert current_config == config_after_original
