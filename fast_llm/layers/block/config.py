import abc
import collections
import functools
import typing
import warnings

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelConfig, LossDef, Preprocessor
from fast_llm.engine.config_utils.parameter import combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.block.block import Block


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


@config_class()
class BaseBlockConfig(BaseModelConfig):
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

    def get_preprocessors(self, distributed_config: DistributedConfig) -> list[Preprocessor]:
        return []

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        return []


@config_class(registry=True)
class BlockConfig(BaseBlockConfig):
    """
    Base configuration class for actual blocks, i.e., base blocks that are also `Layers`.
    """

    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is BlockConfig and cls.get_subclass(default.get("type")) is None:
            from fast_llm.layers.decoder.config import DecoderBlockConfig

            # Default subclass.
            return DecoderBlockConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    @property
    def layer_class(self) -> "type[Block]":
        raise NotImplementedError()

    def get_block(
        self,
        distributed_config: DistributedConfig,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_input: bool = False,
    ) -> "Block":
        return self.layer_class(
            self,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            peft=peft,
            return_input=return_input,
        )


@config_class(registry=True)
class BlockSequenceConfig(BaseModelConfig):
    @classmethod
    def _from_dict(cls, default: dict[str, typing.Any], strict: bool = True) -> typing.Self:
        if cls is BlockSequenceConfig and cls.get_subclass(default.get("type")) is None:
            # Default subclass.
            return FixedBlockSequenceConfig._from_dict(default, strict)
        return super()._from_dict(default, strict=strict)

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> BlockConfig:
        pass

    @abc.abstractmethod
    def get_preprocessors(self, distributed_config: DistributedConfig) -> list[Preprocessor]:
        pass

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        return []


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

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> BlockConfig:
        return self.block

    def get_preprocessors(self, distributed_config: DistributedConfig) -> list[Preprocessor]:
        # TODO: Prevent name conflicts in preprocessed kwargs.
        return self.block.get_preprocessors(distributed_config)

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        return self.block.get_loss_definitions(count=count * self.num_blocks)


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

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> BlockConfig:
        return self.blocks[self.expanded_pattern[index]]

    @functools.cached_property
    def expanded_pattern(self) -> list[str]:
        return (self.pattern * (self.num_blocks // len(self.pattern) + 1))[: self.num_blocks]

    def get_preprocessors(self, distributed_config: DistributedConfig) -> list[Preprocessor]:
        # TODO: Prevent name conflicts in preprocessed kwargs.
        return sum((block.get_preprocessors(distributed_config) for block in self.blocks.values()), [])

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        # TODO: Prevent name conflicts.
        return sum(
            (
                self.blocks[name].get_loss_definitions(count=count * count_)
                for name, count_ in collections.Counter(self.expanded_pattern).items()
            ),
            [],
        )
