import logging
import typing

import torch

from fast_llm.engine.base_model.config import LossDef, ResourceUsageConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.decoder.config import (
    StochasticMixerConfig,
    StochasticMixerKwargs,
    StochasticMixerSamplingStrategy,
)
from fast_llm.logging import get_model_debug_level
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert, safe_merge_dicts

logger = logging.getLogger(__name__)


class StochasticMixer[ConfigType: StochasticMixerConfig](BlockWithBias[ConfigType]):
    """
    A mixer that stochastically samples from multiple mixer options during training.

    In training mode, each forward pass randomly selects one mixer according to
    the sampling strategy. In eval mode, uses the configured inference mixer.

    This is useful for supernet training where you want to train multiple
    architecture variants (e.g., attention vs. Mamba) with different data subsets.
    """

    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_bias: bool = True,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
            return_bias=return_bias,
        )

        # Initialize all mixers
        self.mixers = torch.nn.ModuleDict(
            {
                name: mixer_config.get_layer(
                    distributed_config,
                    hidden_dim,
                    lr_scale=lr_scale,
                    peft=peft,
                    return_bias=return_bias,
                )
                for name, mixer_config in self._config.mixers.items()
            }
        )

        if self._config.sampling_strategy == StochasticMixerSamplingStrategy.full_layout:
            self._sampling_probs = None
        elif self._config.sampling_strategy == StochasticMixerSamplingStrategy.uniform:
            self._sampling_probs = torch.ones(len(self.mixers), device="cpu") / len(self.mixers)
        elif self._config.sampling_strategy == StochasticMixerSamplingStrategy.weighted:
            if self._config.sampling_weights is None:
                raise ValueError("sampling_weights must be provided when using weighted sampling strategy")
            self._sampling_probs = torch.tensor(
                [self._config.sampling_weights[name] for name in self.mixers.keys()],
                dtype=torch.float32,
                device="cpu",
            )
        else:
            raise NotImplementedError(f"Sampling strategy {self._config.sampling_strategy} not implemented")

        logger.info(
            f"Initialized StochasticMixer with {len(self.mixers)} mixers: "
            f"{', '.join(f'{name}={type(mixer).__name__}' for name, mixer in self.mixers.items())} "
            f"(main={self._config.main_mixer_name})"
        )

        # Mark all mixer parameters with allow_no_grad since only one mixer
        # is active per forward pass during training. Even though all mixers
        # will eventually be trained, on any single forward pass, the non-selected
        # mixers won't receive gradients.
        for mixer in self.mixers.values():
            for param in mixer.parameters(recurse=True):
                if hasattr(param, "allow_no_grad"):
                    param.allow_no_grad = True

        # Track mixer selection counts for logging actual proportions during training
        self._mixer_counts_total = {name: 0 for name in self.mixers.keys()}
        self._last_selected_mixer = None

    def setup(self, distributed: Distributed) -> None:
        """Setup all mixers with the distributed context."""
        super().setup(distributed)
        for mixer in self.mixers.values():
            mixer.setup(distributed)

    def _sample_mixer_name(self, kwargs: dict[str, typing.Any]) -> str:
        if not self.training:
            return self._config.main_mixer_name

        # Layout-based selection (full_layout strategy or predefined layout override)
        if StochasticMixerKwargs.layout in kwargs:
            layout = kwargs[StochasticMixerKwargs.layout]
            counter = kwargs[StochasticMixerKwargs.layout_counter]
            idx = counter[0]
            counter[0] += 1
            return layout[idx]

        generator = kwargs[StochasticMixerKwargs.generator]
        mixer_idx = torch.multinomial(self._sampling_probs, num_samples=1, generator=generator).item()
        return list(self.mixers.keys())[mixer_idx]

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mixer_name = self._sample_mixer_name(kwargs)

        if self.training:
            self._mixer_counts_total[mixer_name] += 1
            self._last_selected_mixer = mixer_name

            if metrics is not None:
                # Use module_name as prefix to distinguish between different layer indices
                metric_prefix = f"{self.module_name}/stochastic"

                # Instantaneous metric: last selected mixer
                metrics[f"{metric_prefix}/last_selected"] = mixer_name

                # Cumulative metrics: total counts and proportions
                total_count = sum(self._mixer_counts_total.values())
                for name, count in self._mixer_counts_total.items():
                    proportion = count / total_count if total_count > 0 else 0.0
                    metrics[f"{metric_prefix}/{name}_count_total"] = count
                    metrics[f"{metric_prefix}/{name}_proportion_total"] = proportion

        if get_model_debug_level() > 0:
            from fast_llm.layers.block.config import BlockKwargs

            iteration = kwargs.get(BlockKwargs.iteration, "?")
            logger.info(
                f"StochasticMixer iter={iteration} selecting mixer '{mixer_name}' "
                f"({type(self.mixers[mixer_name]).__name__})"
            )

        return self.mixers[mixer_name]._forward(input_, kwargs, losses, metrics)

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return safe_merge_dicts(*(mixer.get_preprocessing_config() for mixer in self.mixers.values()))

    def _sample_allocation(self, num_layers: int, generator: torch.Generator) -> list[int]:
        """
        Sample a composition of num_layers into num_mixers bins uniformly.

        Uses stars-and-bars: picks (M-1) bar positions from {0, ..., N+M-2},
        giving each mixer a count. All integer partitions are equally likely.
        """
        M = len(self.mixers)
        N = num_layers
        if M == 1:
            return [N]
        bars = torch.randperm(N + M - 1, generator=generator)[: M - 1].sort().values
        padded = torch.cat([torch.tensor([-1]), bars, torch.tensor([N + M - 1])])
        counts = (padded[1:] - padded[:-1] - 1).tolist()
        return counts

    def _sample_placement(self, counts: list[int], num_layers: int, generator: torch.Generator) -> list[str]:
        """
        Given per-mixer counts, create a shuffled layout.
        """
        mixer_names = list(self.mixers.keys())
        layout = []
        for name, count in zip(mixer_names, counts):
            layout.extend([name] * count)
        perm = torch.randperm(num_layers, generator=generator)
        return [layout[i] for i in perm.tolist()]

    def _sample_predefined_layout(self, num_layers: int, generator: torch.Generator) -> list[str] | None:
        """
        With probability `predefined_layout_probability`, pick a predefined layout uniformly.
        Returns None if we should use the normal sampling strategy instead.
        """
        if not self._config.predefined_layouts or self._config.predefined_layout_probability <= 0:
            return None
        coin = torch.rand(1, generator=generator).item()
        if coin >= self._config.predefined_layout_probability:
            return None
        idx = torch.randint(len(self._config.predefined_layouts), (1,), generator=generator).item()
        layout = list(self._config.predefined_layouts[idx])
        Assert.eq(len(layout), num_layers)
        return layout

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        from fast_llm.engine.distributed.config import MAX_SEED
        from fast_llm.layers.block.config import BlockKwargs

        iteration = kwargs[BlockKwargs.iteration]
        generator = torch.Generator(device="cpu")
        seed = (self._distributed_config.seed + self._config.seed_shift + iteration) % MAX_SEED
        generator.manual_seed(seed)
        kwargs[StochasticMixerKwargs.generator] = generator

        num_layers = kwargs[BlockKwargs.num_blocks_in_sequence]
        predefined = self._sample_predefined_layout(num_layers, generator)

        if predefined is not None:
            # Use predefined layout (overrides any sampling strategy)
            kwargs[StochasticMixerKwargs.layout] = predefined
            kwargs[StochasticMixerKwargs.layout_counter] = [0]
        elif self._config.sampling_strategy == StochasticMixerSamplingStrategy.full_layout:
            counts = self._sample_allocation(num_layers, generator)
            layout = self._sample_placement(counts, num_layers, generator)
            kwargs[StochasticMixerKwargs.layout] = layout
            kwargs[StochasticMixerKwargs.layout_counter] = [0]

        for mixer in self.mixers.values():
            mixer.preprocess(kwargs)

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        """
        Return expected compute usage (weighted average of all mixers).

        This gives a more accurate estimate than just using one mixer,
        since during training we'll be using all of them according to
        their sampling probabilities.
        """
        usages = [mixer.get_compute_usage(input_, kwargs, config) for mixer in self.mixers.values()]

        if self._sampling_probs is not None:
            # Weight by sampling probability and return the expected value
            expected_usage = sum(usage * prob.item() for usage, prob in zip(usages, self._sampling_probs))
        else:
            # full_layout: uniform over compositions, so equal expected weight per mixer
            expected_usage = sum(usages) / len(usages)

        return int(expected_usage)

    def get_loss_definitions(self) -> list[LossDef]:
        """
        Merge loss definitions from all mixers with namespacing.

        Each mixer's losses are namespaced with the mixer name to avoid conflicts.
        This ensures we allocate space for any auxiliary losses that any
        of the mixers might need, even if multiple mixers have losses with the same name.
        """
        all_losses = []
        for mixer_name, mixer in self.mixers.items():
            mixer_losses = mixer.get_loss_definitions()
            # Namespace each loss with the mixer name to avoid conflicts
            for loss_def in mixer_losses:
                namespaced_loss = LossDef(
                    name=f"{mixer_name}/{loss_def.name}",
                    dtype=loss_def.dtype,
                )
                all_losses.append(namespaced_loss)

        return all_losses
