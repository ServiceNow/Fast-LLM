import copy
import json
import logging
import pathlib

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.run import log_main_rank
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.evaluation.evaluator import TrainingProgress
from fast_llm.engine.training.config import TrainerConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.decoder.config import StochasticMixerConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


@config_class()
class BeamSearchConfig(RunnableConfig):
    """
    Hierarchical beam search for finding optimal mixer placement in a supernet.

    The mixers in the stochastic mixer config are ranked by their order:
    - mixers[0] is primary (highest quality, most expensive)
    - mixers[1] is secondary (medium quality, medium cost)
    - mixers[2] is tertiary (lowest cost)
    - etc.

    The algorithm works hierarchically:
    1. Phase 1: Find best placement for budgets[0] primary mixer layers
       (non-primary layers use secondary as baseline)
    2. Phase 2: Given fixed primary positions, find best placement for budgets[1] secondary layers
       (non-secondary layers use tertiary as baseline)
    3. Continue for additional levels if specified

    Example: With FA/SWA/LA and budgets=[4, 8]:
    - Find best 4 layers for FA (others use SWA during evaluation)
    - Given those 4 FA layers, find best 8 layers for SWA (others use LA)
    - Remaining layers use LA
    """

    training_config: pathlib.Path = Field(
        desc="Path to the training config with supernet checkpoint.",
        hint=FieldHint.core,
    )

    budgets: list[int] = Field(
        desc="Budget for each mixer level. budgets[i] specifies how many layers use mixers[i]. "
        "Length must be less than number of mixers (last mixer is used for all remaining layers).",
        hint=FieldHint.core,
    )

    beam_width: int = Field(
        default=12,
        desc="Number of top candidates to keep at each growth step (8-16 recommended).",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )

    initial_beam_width: int = Field(
        default=12,
        desc="Number of top single-layer configs to seed each beam phase (8-16 recommended).",
        hint=FieldHint.feature,
        valid=check_field(Assert.gt, 0),
    )

    output_path: pathlib.Path = Field(
        desc="Path to save beam search results.",
        hint=FieldHint.core,
    )

    early_stop_threshold: float = Field(
        default=0.001,
        desc="Stop growth phase if best score improvement is below this threshold.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 0),
    )

    score_metric: str = Field(
        desc="Name of the metric to use as the optimization score. "
        "Should match the format 'evaluator_name/metric_name' from evaluation results.",
        hint=FieldHint.core,
    )

    higher_is_better: bool = Field(
        default=True,
        desc="Whether higher metric values are better. Set to False for metrics like loss.",
        hint=FieldHint.feature,
    )

    output_checkpoint_path: pathlib.Path | None = Field(
        default=None,
        desc="Path to save the best configuration as a converted checkpoint. " "If None, only JSON results are saved.",
        hint=FieldHint.feature,
    )

    def run(self) -> None:
        log_main_rank("Loading base training config...")
        base_config = self._load_training_config()

        num_layers = self._get_num_layers(base_config)
        num_mixers = self._get_num_mixers(base_config)

        Assert.lt(len(self.budgets), num_mixers)
        for budget in self.budgets:
            Assert.gt(budget, 0)
        Assert.leq(sum(self.budgets), num_layers)

        log_main_rank(f"\n{'='*60}")
        log_main_rank(f"Hierarchical Beam Search Configuration")
        log_main_rank(f"{'='*60}")
        log_main_rank(f"Total layers: {num_layers}")
        log_main_rank(f"Number of mixer types: {num_mixers}")
        log_main_rank(f"Budgets: {self.budgets}")
        log_main_rank(f"Beam width: {self.beam_width}")
        log_main_rank(f"Initial beam width: {self.initial_beam_width}")

        self._validate_stochastic_mixer(base_config, num_layers)

        log_main_rank("\nInitializing evaluation infrastructure...")
        self._setup_evaluation(base_config)

        # Run beam search inside the Run context manager
        with self._run:
            layer_assignments = {}
            phase_results = []

            for phase_idx, budget in enumerate(self.budgets):
                phase_result = self._run_beam_search_phase(
                    base_config, num_layers, phase_idx, budget, layer_assignments
                )
                phase_results.append(phase_result)

                for layer_idx in phase_result["best_layers"]:
                    layer_assignments[layer_idx] = phase_idx

            # Assign remaining layers to the last mixer
            self._assign_remaining_layers(layer_assignments, num_layers, len(self.budgets))

            # Final evaluation
            log_main_rank(f"\n{'='*60}")
            log_main_rank(f"FINAL EVALUATION")
            log_main_rank(f"{'='*60}")

            final_score = self._evaluate_assignment(base_config, layer_assignments, num_layers)

            log_main_rank(f"Final configuration:")
            for mixer_idx in range(num_mixers):
                layers = [l for l, m in layer_assignments.items() if m == mixer_idx]
                log_main_rank(f"  mixer[{mixer_idx}]: {len(layers)} layers - {sorted(layers)}")
            log_main_rank(f"Final score: {final_score:.4f}")

            self._save_results(phase_results, layer_assignments, final_score, num_layers, num_mixers)

            if self.output_checkpoint_path is not None:
                log_main_rank(f"\n{'='*60}")
                log_main_rank(f"Converting best configuration to checkpoint")
                log_main_rank(f"{'='*60}")
                self._save_best_checkpoint(base_config, layer_assignments, num_layers)

    def _run_beam_search_phase(
        self,
        base_config: TrainerConfig,
        num_layers: int,
        phase_idx: int,
        budget: int,
        fixed_assignments: dict[int, int],
    ) -> dict:
        """Run one phase of hierarchical beam search."""
        mixer_idx = phase_idx
        next_mixer_idx = phase_idx + 1

        log_main_rank(f"\n{'='*60}")
        log_main_rank(f"PHASE {phase_idx + 1}: Optimizing placement for mixer[{mixer_idx}]")
        log_main_rank(f"Budget: {budget} layers")
        log_main_rank(f"Baseline for non-assigned layers: mixer[{next_mixer_idx}]")
        log_main_rank(f"{'='*60}")

        unassigned_layers = [idx for idx in range(num_layers) if idx not in fixed_assignments]
        log_main_rank(f"Unassigned layers: {len(unassigned_layers)} out of {num_layers}")

        # Pre-score individual layers
        layer_scores = self._prescore_layers(
            base_config, num_layers, mixer_idx, next_mixer_idx, unassigned_layers, fixed_assignments
        )

        # Seed and grow beam
        beam = self._grow_beam(
            base_config,
            num_layers,
            mixer_idx,
            next_mixer_idx,
            budget,
            unassigned_layers,
            fixed_assignments,
            layer_scores,
        )

        log_main_rank(f"\nPhase {phase_idx + 1} complete!")
        log_main_rank(f"Best layers for mixer[{mixer_idx}]: {beam[0]['layers']}")
        log_main_rank(f"Best score: {beam[0]['score']:.4f}")

        return {
            "best_layers": beam[0]["layers"],
            "best_score": beam[0]["score"],
            "beam": beam,
            "layer_scores": layer_scores,
        }

    def _prescore_layers(
        self,
        base_config: TrainerConfig,
        num_layers: int,
        mixer_idx: int,
        baseline_mixer_idx: int,
        unassigned_layers: list[int],
        fixed_assignments: dict[int, int],
    ) -> list[tuple[int, float]]:
        """Pre-score individual layers to seed the beam."""
        log_main_rank(f"\nPre-scoring unassigned layers...")

        layer_scores = []
        for layer_idx in unassigned_layers:
            assignment = self._create_test_assignment(
                fixed_assignments, [layer_idx], mixer_idx, unassigned_layers, baseline_mixer_idx
            )
            score = self._evaluate_assignment(base_config, assignment, num_layers)
            layer_scores.append((layer_idx, score))
            log_main_rank(f"  Layer {layer_idx}: {score:.4f}")

        layer_scores.sort(key=lambda x: x[1], reverse=self.higher_is_better)

        log_main_rank(f"\nLayer ranking for mixer[{mixer_idx}]:")
        for rank, (layer_idx, score) in enumerate(layer_scores[:10]):
            log_main_rank(f"  {rank+1}. Layer {layer_idx}: {score:.4f}")

        return layer_scores

    def _grow_beam(
        self,
        base_config: TrainerConfig,
        num_layers: int,
        mixer_idx: int,
        baseline_mixer_idx: int,
        budget: int,
        unassigned_layers: list[int],
        fixed_assignments: dict[int, int],
        layer_scores: list[tuple[int, float]],
    ) -> list[dict]:
        """Grow the beam from seed to budget size."""
        log_main_rank(f"\nSeeding beam with top {self.initial_beam_width} layers...")

        beam = [
            {"layers": [layer_idx], "score": score} for layer_idx, score in layer_scores[: self.initial_beam_width]
        ]

        log_main_rank(f"\nGrowing beam to budget of {budget}...")
        best_score = beam[0]["score"]

        for growth_step in range(1, budget):
            log_main_rank(f"\nGrowth step {growth_step}: Adding layer #{growth_step+1}")

            candidates = self._generate_candidates(beam, unassigned_layers)
            log_main_rank(f"Generated {len(candidates)} unique candidates")

            self._evaluate_candidates(
                candidates,
                base_config,
                num_layers,
                mixer_idx,
                baseline_mixer_idx,
                unassigned_layers,
                fixed_assignments,
            )

            candidates.sort(key=lambda x: x["score"], reverse=self.higher_is_better)
            beam = candidates[: self.beam_width]

            self._log_top_candidates(beam)

            new_best_score = beam[0]["score"]
            if self._should_early_stop(best_score, new_best_score):
                break
            best_score = new_best_score

        return beam

    def _generate_candidates(self, beam: list[dict], unassigned_layers: list[int]) -> list[dict]:
        """Generate new candidates by expanding each beam entry."""
        candidates = []
        seen_candidates = set()

        for beam_candidate in beam:
            existing_layers = set(beam_candidate["layers"])

            for layer_idx in unassigned_layers:
                if layer_idx in existing_layers:
                    continue

                new_layers = tuple(sorted(beam_candidate["layers"] + [layer_idx]))

                if new_layers in seen_candidates:
                    continue
                seen_candidates.add(new_layers)

                candidates.append({"layers": list(new_layers), "score": None})

        return candidates

    def _evaluate_candidates(
        self,
        candidates: list[dict],
        base_config: TrainerConfig,
        num_layers: int,
        mixer_idx: int,
        baseline_mixer_idx: int,
        unassigned_layers: list[int],
        fixed_assignments: dict[int, int],
    ) -> None:
        """Evaluate all candidates and store scores."""
        for i, candidate in enumerate(candidates):
            assignment = self._create_test_assignment(
                fixed_assignments, candidate["layers"], mixer_idx, unassigned_layers, baseline_mixer_idx
            )
            candidate["score"] = self._evaluate_assignment(base_config, assignment, num_layers)

            if (i + 1) % max(1, len(candidates) // 10) == 0:
                log_main_rank(f"  Evaluated {i+1}/{len(candidates)} candidates...")

    def _create_test_assignment(
        self,
        fixed_assignments: dict[int, int],
        target_layers: list[int],
        target_mixer_idx: int,
        unassigned_layers: list[int],
        baseline_mixer_idx: int,
    ) -> dict[int, int]:
        """Create a test assignment for evaluation."""
        assignment = fixed_assignments.copy()

        for layer_idx in target_layers:
            assignment[layer_idx] = target_mixer_idx

        for layer_idx in unassigned_layers:
            if layer_idx not in assignment:
                assignment[layer_idx] = baseline_mixer_idx

        return assignment

    def _log_top_candidates(self, beam: list[dict]) -> None:
        """Log the top candidates in the beam."""
        log_main_rank(f"\nTop {min(3, len(beam))} candidates:")
        for i, candidate in enumerate(beam[:3]):
            log_main_rank(f"  {i+1}. {candidate['layers']} - Score: {candidate['score']:.4f}")

    def _should_early_stop(self, best_score: float, new_best_score: float) -> bool:
        """Check if early stopping criteria is met."""
        improvement = (new_best_score - best_score) if self.higher_is_better else (best_score - new_best_score)

        if improvement < self.early_stop_threshold:
            log_main_rank(f"Early stopping: improvement {improvement:.4f} < threshold {self.early_stop_threshold}")
            return True
        return False

    def _assign_remaining_layers(
        self, layer_assignments: dict[int, int], num_layers: int, last_mixer_idx: int
    ) -> None:
        """Assign all remaining unassigned layers to the last mixer."""
        for layer_idx in range(num_layers):
            if layer_idx not in layer_assignments:
                layer_assignments[layer_idx] = last_mixer_idx

    def _validate_stochastic_mixer(self, base_config: TrainerConfig, num_layers: int) -> None:
        """Validate that all layers use StochasticMixerConfig."""
        decoder_config = self._get_decoder_config(base_config)

        if type(decoder_config) is FixedBlockSequenceConfig:
            if not isinstance(decoder_config.block.mixer, StochasticMixerConfig):
                raise ValueError(
                    f"All decoder blocks must use StochasticMixerConfig. "
                    f"Found: {type(decoder_config.block.mixer).__name__}"
                )
        elif type(decoder_config) is PatternBlockSequenceConfig:
            for block in decoder_config.pattern_blocks:
                if not isinstance(block.block.mixer, StochasticMixerConfig):
                    raise ValueError(
                        f"All decoder blocks must use StochasticMixerConfig. "
                        f"Found: {type(block.block.mixer).__name__}"
                    )
        else:
            raise NotImplementedError(f"Unknown decoder config type: {type(decoder_config).__name__}")

        log_main_rank(f"Validated: All {num_layers} layers use StochasticMixerConfig")

    def _setup_evaluation(self, base_config: TrainerConfig) -> None:
        """Setup evaluation infrastructure once and reuse across all evaluations."""
        self._eval_base_config = self._create_eval_base_config(base_config)
        self._distributed = Distributed(self._eval_base_config.model.distributed)
        self._run = self._eval_base_config.get_run(self._distributed)
        self._trainer = self._eval_base_config.get_trainer_class()(config=self._eval_base_config)
        self._trainer.setup(self._distributed, self._run)

        log_main_rank("Evaluation infrastructure ready")

    def _evaluate_assignment(
        self,
        base_config: TrainerConfig,
        layer_assignments: dict[int, int],
        num_layers: int,
    ) -> float:
        """Evaluate a complete layer-to-mixer assignment."""
        self._update_model_architecture(layer_assignments, num_layers)

        metrics = {}

        self._trainer._evaluator_runner.run(
            metrics=metrics,
            training_progress=TrainingProgress(
                done=True,
                completed_steps=self._trainer._completed_steps,
                consumed_samples=self._trainer._consumed_samples,
                consumed_tokens=self._trainer._consumed_tokens,
            ),
        )

        if self.score_metric not in metrics:
            raise ValueError(
                f"Score metric '{self.score_metric}' not found in evaluation results. "
                f"Available metrics: {list(metrics.keys())}"
            )

        score = metrics[self.score_metric]
        logger.debug(f"Evaluation score ({self.score_metric}): {score}")

        return score

    def _update_model_architecture(self, layer_assignments: dict[int, int], num_layers: int) -> None:
        """Update the model architecture in-place by modifying main_mixer_index."""
        base_model = self._trainer._multi_stage.base_model
        self._trainer._multi_stage.eval()

        decoder = base_model.decoder

        for layer_idx in range(num_layers):
            mixer_idx = layer_assignments[layer_idx]
            decoder[layer_idx].mixer._config.main_mixer_index = mixer_idx

    def _create_eval_base_config(self, base_config: TrainerConfig) -> TrainerConfig:
        """Create base evaluation config (train_iters=0)."""

        config_dict = base_config.to_dict()
        config_dict["training"]["train_iters"] = 0

        return TrainerConfig.from_dict(config_dict)

    def _save_best_checkpoint(
        self, base_config: TrainerConfig, layer_assignments: dict[int, int], num_layers: int
    ) -> None:
        """Save the best configuration as a converted checkpoint."""
        import yaml

        config_dict = base_config.to_dict()
        model_config_dict = config_dict["model"]["base_model"]
        decoder_config = self._get_decoder_config(base_config)

        # Get base block dict
        if type(decoder_config) is FixedBlockSequenceConfig:
            base_block_dict = model_config_dict["decoder"]["block"]
        elif type(decoder_config) is PatternBlockSequenceConfig:
            base_block_dict = model_config_dict["decoder"]["pattern_blocks"][0]["block"]
        else:
            raise NotImplementedError(f"Unknown decoder config type: {type(decoder_config).__name__}")

        # Create pattern_blocks with layer-specific mixer assignments
        pattern_blocks = []
        for layer_idx in range(num_layers):
            block_dict = copy.deepcopy(base_block_dict)
            block_dict["mixer"]["main_mixer_index"] = layer_assignments[layer_idx]
            pattern_blocks.append({"block": block_dict, "repeat": 1})

        # Convert to pattern_blocks format
        model_config_dict["decoder"]["pattern_blocks"] = pattern_blocks
        model_config_dict["decoder"].pop("num_blocks", None)
        model_config_dict["decoder"].pop("block", None)
        model_config_dict["decoder"].pop("blocks", None)
        model_config_dict["decoder"].pop("pattern", None)

        config_output_path = self.output_checkpoint_path.parent / "best_config.yaml"
        config_output_path.parent.mkdir(parents=True, exist_ok=True)

        with config_output_path.open("w") as f:
            yaml.safe_dump(config_dict, f)

        log_main_rank(f"Saved best configuration to {config_output_path}")
        log_main_rank("Checkpoint conversion not yet implemented. Only the configuration has been saved.")

    def _load_training_config(self) -> TrainerConfig:
        """Load the training configuration from the provided path."""
        import yaml

        config_dict = yaml.safe_load(self.training_config.open("r"))
        return TrainerConfig.from_dict(config_dict)

    def _get_decoder_config(self, config: TrainerConfig):
        """Get the decoder config from training config."""
        return config.model.base_model.decoder

    def _get_num_layers(self, config: TrainerConfig) -> int:
        """Get the number of decoder layers."""
        decoder_config = self._get_decoder_config(config)

        if type(decoder_config) is PatternBlockSequenceConfig:
            return sum(block.repeat for block in decoder_config.pattern_blocks)
        elif type(decoder_config) is FixedBlockSequenceConfig:
            return decoder_config.num_blocks
        else:
            raise NotImplementedError(f"Unknown decoder config type: {type(decoder_config).__name__}")

    def _get_num_mixers(self, config: TrainerConfig) -> int:
        """Get the number of mixer options in the stochastic mixer."""
        decoder_config = self._get_decoder_config(config)

        if type(decoder_config) is FixedBlockSequenceConfig:
            mixer_config = decoder_config.block.mixer
        elif type(decoder_config) is PatternBlockSequenceConfig:
            mixer_config = decoder_config.pattern_blocks[0].block.mixer
        else:
            raise NotImplementedError(f"Unknown decoder config type: {type(decoder_config).__name__}")

        Assert.custom(isinstance, mixer_config, StochasticMixerConfig)
        return len(mixer_config.mixers)

    def _save_results(
        self,
        phase_results: list[dict],
        layer_assignments: dict[int, int],
        final_score: float,
        num_layers: int,
        num_mixers: int,
    ) -> None:
        """Save beam search results to file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "config": {
                "num_layers": num_layers,
                "num_mixers": num_mixers,
                "budgets": self.budgets,
                "beam_width": self.beam_width,
                "initial_beam_width": self.initial_beam_width,
            },
            "phases": [
                {
                    "mixer_index": i,
                    "budget": self.budgets[i],
                    "best_layers": phase["best_layers"],
                    "best_score": phase["best_score"],
                    "pre_scoring": [
                        {"layer": layer_idx, "score": score} for layer_idx, score in phase["layer_scores"]
                    ],
                }
                for i, phase in enumerate(phase_results)
            ],
            "final_configuration": {
                "layer_assignments": {str(k): v for k, v in layer_assignments.items()},
                "score": final_score,
                "summary": {
                    f"mixer[{mixer_idx}]": sorted([l for l, m in layer_assignments.items() if m == mixer_idx])
                    for mixer_idx in range(num_mixers)
                },
            },
        }

        with self.output_path.open("w") as f:
            json.dump(results, f, indent=2)

        log_main_rank(f"\nResults saved to {self.output_path}")


if __name__ == "__main__":
    BeamSearchConfig.parse_and_run()
