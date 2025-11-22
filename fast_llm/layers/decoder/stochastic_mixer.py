import logging
import typing

import torch

from fast_llm.engine.base_model.config import LossDef, ResourceUsageConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.decoder.block import BlockWithBias
from fast_llm.layers.decoder.config import StochasticMixerConfig, StochasticMixerKwargs, StochasticMixerSamplingStrategy
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
        self.mixers = torch.nn.ModuleDict(
            {
                name: mixer_config.get_layer(
                    distributed_config,
                    hidden_dim,
                    lr_scale,
                    peft=peft,
                    return_bias=return_bias,
                )
                for name, mixer_config in self._config.mixers.items()
            }
        )

        if self._config.sampling_strategy == StochasticMixerSamplingStrategy.uniform:
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

    def setup(self, distributed: Distributed) -> None:
        """Setup all mixers with the distributed context."""
        super().setup(distributed)
        for mixer in self.mixers.values():
            mixer.setup(distributed)

    def _sample_mixer_name(self, kwargs: dict[str, typing.Any]) -> str:
        if not self.training:
            return self._config.main_mixer_name

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

        if self._debug.enabled:
            logger.debug(f"StochasticMixer selecting mixer {mixer_name}: {type(self.mixers[mixer_name]).__name__}")

        return self.mixers[mixer_name]._forward(input_, kwargs, losses, metrics)

    def preprocess(self, batch: torch.Tensor, kwargs: dict[str, typing.Any]) -> None:
        from fast_llm.layers.block.config import BlockKwargs

        iteration = kwargs[BlockKwargs.iteration]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(iteration)
        kwargs[StochasticMixerKwargs.generator] = generator

        for mixer in self.mixers.values():
            mixer.preprocess(batch, kwargs)

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        """
        Return expected compute usage (weighted average of all mixers).

        This gives a more accurate estimate than just using one mixer,
        since during training we'll be using all of them according to
        their sampling probabilities.
        """
        usages = [mixer.get_compute_usage(input_, kwargs, config) for mixer in self.mixers.values()]

        # Weight by sampling probability and return the expected value
        expected_usage = sum(usage * prob.item() for usage, prob in zip(usages, self._sampling_probs))

        return int(expected_usage)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        """
        Merge loss definitions from all mixers with namespacing.

        Each mixer's losses are namespaced with the mixer name to avoid conflicts.
        This ensures we allocate space for any auxiliary losses that any
        of the mixers might need, even if multiple mixers have losses with the same name.
        """
        all_losses = []
        for mixer_name, mixer in self.mixers.items():
            mixer_losses = mixer.get_loss_definitions(count=count)
            # Namespace each loss with the mixer name to avoid conflicts
            for loss_def in mixer_losses:
                namespaced_loss = LossDef(
                    name=f"{mixer_name}/{loss_def.name}",
                    formatted_name=f"{mixer_name}/{loss_def.formatted_name}",
                    count=loss_def.count,
                    dtype=loss_def.dtype,
                )
                all_losses.append(namespaced_loss)

        return all_losses
