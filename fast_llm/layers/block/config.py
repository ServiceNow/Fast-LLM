import functools
import logging
import typing
import warnings

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import ModuleConfig
from fast_llm.engine.config_utils.parameter import combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.utils import Assert, log

if typing.TYPE_CHECKING:
    from fast_llm.layers.block.block import BlockBase
    from fast_llm.layers.block.sequence import FixedBlockSequence, PatternBlockSequence

logger = logging.getLogger(__name__)


class BlockDimNames:
    # A set of common tensor dim names packed into a namespace.
    # Input dimensions (variable)
    # TODO: Does batch belong here?
    batch = "batch"
    # TODO: Distinguish micro-sequence?
    sequence_q = "sequence_q"
    sequence_q_tp = "sequence_q_tp"
    sequence_k = "sequence_k"
    hidden = "hidden"


class BlockKwargs:
    sequence_first = "sequence_first"
    hidden_dims = "hidden_dims"
    sequence_q_dim = "sequence_q_dim"
    sequence_k_dim = "sequence_k_dim"
    # TODO: These are confusing
    sequence_length = "sequence_length"
    sequence_lengths = "sequence_lengths"
    # TODO: Belongs elsewhere?
    grad_output = "grad_output"
    activation_distillation_targets = "activation_distillation_targets"
    iteration = "iteration"
    device = "device"
    hidden_states = "hidden_states"
    output_hidden_states = "output_hidden_states"
    padding_mask = "padding_mask"


@config_class(registry=True)
class BlockConfig(ModuleConfig):
    """
    Base configuration class for blocks and block-like layers (mlp, mixers, etc.).
    """

    _abstract = True

    lr_scale: float | None = Field(
        default=None,
        desc="Scaling factor for the layer learning rate."
        " Combines multiplicatively with the scale set by the parent and child layers, if applicable.",
        hint=FieldHint.feature,
    )

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is BlockConfig and cls.get_subclass(default.get("type")) is None:
            from fast_llm.layers.decoder.config import DecoderBlockConfig

            # Default subclass.
            return DecoderBlockConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    @property
    def layer_class(self) -> "type[BlockBase]":
        raise NotImplementedError()

    def get_layer(
        self,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        *,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "BlockBase":
        return self.layer_class(
            self,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
        )

    def get_distillation_models(self) -> set[str]:
        return set()


@config_class(registry=True)
class BlockSequenceConfig(BlockConfig):
    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is BlockSequenceConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return FixedBlockSequenceConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)


@config_class(dynamic_type={BlockSequenceConfig: "fixed"})
class FixedBlockSequenceConfig(BlockSequenceConfig):
    _abstract = False
    block: BlockConfig = Field(
        desc="Common configuration for all the blocks.",
        hint=FieldHint.architecture,
    )
    num_blocks: int = Field(
        default=12,
        desc="Number of blocks in the model.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.geq, 0),
    )

    @property
    def layer_class(self) -> "type[FixedBlockSequence]":
        from fast_llm.layers.block.sequence import FixedBlockSequence

        return FixedBlockSequence

    def get_distillation_models(self) -> set[str]:
        return self.block.get_distillation_models()


@config_class(dynamic_type={BlockSequenceConfig: "pattern"})
class PatternBlockSequenceConfig(BlockSequenceConfig):
    _abstract = False
    blocks: dict[str, BlockConfig] = Field()
    pattern: list[str] = Field(
        default=None,
        desc="The name of each block (key in `blocks`) in the repeated pattern.",
        hint=FieldHint.architecture,
    )
    num_blocks: int = Field(
        default=12,
        desc="Number of blocks in the model.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.geq, 0),
    )

    def _validate(self):
        if not self.blocks:
            raise ValueError("No block configuration provided")
        if not self.pattern:
            raise ValueError("No block pattern provided")
        used_blocks = set(self.pattern)
        available_blocks = set(self.blocks)
        if missing := used_blocks - available_blocks:
            raise ValueError(f"The following blocks are present in the pattern but undefined: {missing}")
        if extra := available_blocks - used_blocks:
            raise warnings.warn(f"The following blocks are defined but unused: {extra}")

        super()._validate()

    @property
    def layer_class(self) -> "type[PatternBlockSequence]":
        from fast_llm.layers.block.sequence import PatternBlockSequence

        return PatternBlockSequence

    @functools.cached_property
    def expanded_pattern(self) -> list[str]:
        # The complete list of block names, expanded to `num_blocks`
        return (self.pattern * (self.num_blocks // len(self.pattern) + 1))[: self.num_blocks]

    @functools.cached_property
    def preprocessing_layers(self) -> dict[str, int]:
        # The index at which each block first appears. These blocks are used for preprocessing.
        return {name: self.expanded_pattern.index(name) for name in set(self.expanded_pattern)}

    def get_distillation_models(self) -> set[str]:
        models = set()
        for block in self.blocks.values():
            models.update(block.get_distillation_models())
        return models

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        # Patch creeping type parameters from pretrained model
        # TODO: fix this
        if "block" in default:
            removed = default.pop("block")
            log(
                f"Removing 'block' from default dict in PatternBlockSequenceConfig._from_dict: {removed}",
                log_fn=logger.warning,
            )
        return super()._from_dict(default, strict=strict)
