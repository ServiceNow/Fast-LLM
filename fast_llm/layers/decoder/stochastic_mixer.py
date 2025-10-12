import logging
import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.engine.base_model.config import LossDef, ResourceUsageConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.decoder.config import SamplingStrategy, StochasticMixerConfig
from fast_llm.tensor import TensorMeta

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
        self.mixers = torch.nn.ModuleList(
            [
                mixer_config.get_layer(
                    distributed_config,
                    hidden_dim,
                    lr_scale,
                    peft=peft,
                    return_bias=return_bias,
                )
                for mixer_config in self._config.mixers
            ]
        )

        # Precompute sampling probabilities as a tensor
        if self._config.sampling_strategy == SamplingStrategy.uniform:
            self._sampling_probs = torch.ones(len(self.mixers)) / len(self.mixers)
        elif self._config.sampling_strategy == SamplingStrategy.weighted:
            if self._config.sampling_weights is None:
                raise ValueError("sampling_weights must be provided when using weighted sampling strategy")
            self._sampling_probs = torch.tensor(self._config.sampling_weights, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Sampling strategy {self._config.sampling_strategy} not implemented")

        logger.info(
            f"Initialized StochasticMixer with {len(self.mixers)} mixers: "
            f"{[type(m).__name__ for m in self.mixers]}"
        )

    def setup(self, distributed: Distributed) -> None:
        """Setup all mixers with the distributed context."""
        super().setup(distributed)
        for mixer in self.mixers:
            mixer.setup(distributed)

    def _sample_mixer_index(self) -> int:
        """
        Sample a mixer index according to the configured strategy.

        Returns:
            Index of the mixer to use for this forward pass.
        """
        if not self.training:
            # Inference mode: use the configured main mixer
            return self._config.main_mixer_index

        # Training mode: stochastic sampling
        # Use distributed RNG to ensure consistency across TP/PP ranks
        # This ensures all ranks in a TP/PP group use the same mixer
        generator = self._distributed.tp_generator if self._sequence_parallel else self._distributed.pp_generator

        with set_generator(generator):
            # Sample from categorical distribution
            idx = torch.multinomial(self._sampling_probs, num_samples=1).item()

        return idx

    def _forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through a randomly selected mixer.

        Args:
            input_: Input tensor
            kwargs: Forward pass arguments
            losses: Optional dictionary to store losses
            metrics: Optional dictionary to store metrics

        Returns:
            Tuple of (output tensor, bias tensor or None)
        """
        # Sample which mixer to use
        mixer_idx = self._sample_mixer_index()

        if self._debug.enabled:
            logger.debug(f"StochasticMixer selecting mixer {mixer_idx}: {type(self.mixers[mixer_idx]).__name__}")

        # Forward through selected mixer
        return self.mixers[mixer_idx]._forward(input_, kwargs, losses, metrics)

    def preprocess(self, batch: torch.Tensor, kwargs: dict[str, typing.Any]) -> None:
        """
        Preprocess for all mixers.

        Since we don't know which mixer will be selected during training,
        we need to preprocess for all of them. This includes things like
        attention masks, rotary embeddings, etc.
        """
        for mixer in self.mixers:
            mixer.preprocess(batch, kwargs)

    def get_compute_usage(
        self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig
    ) -> int:
        """
        Return expected compute usage (weighted average of all mixers).

        This gives a more accurate estimate than just using one mixer,
        since during training we'll be using all of them according to
        their sampling probabilities.
        """
        usages = [mixer.get_compute_usage(input_, kwargs, config) for mixer in self.mixers]

        # Weight by sampling probability and return the expected value
        expected_usage = sum(usage * prob.item() for usage, prob in zip(usages, self._sampling_probs))

        return int(expected_usage)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        """
        Merge loss definitions from all mixers.

        Returns the union of all loss definitions, deduplicated by name.
        This ensures we allocate space for any auxiliary losses that any
        of the mixers might need.
        """
        all_losses = []
        for mixer in self.mixers:
            all_losses.extend(mixer.get_loss_definitions(count=count))

        # Deduplicate by loss name
        seen = set()
        unique_losses = []
        for loss_def in all_losses:
            if loss_def.name not in seen:
                seen.add(loss_def.name)
                unique_losses.append(loss_def)

        return unique_losses
