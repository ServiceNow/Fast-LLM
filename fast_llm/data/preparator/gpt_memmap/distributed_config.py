import os
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.utils import Assert


@config_class
class DatasetPreparatorDistributedConfig(Config):
    # TODO: Unify with fast_llm.engine.distributed.config.DistributedConfig

    default_world_size: typing.ClassVar[int] = int(os.environ.get("WORLD_SIZE", 1))
    default_rank: typing.ClassVar[int] = int(os.environ.get("RANK", 0))
    world_size: int = Field(
        default=None,
        desc="Size of the world group. Typically provided by torchrun or equivalent through the `WORLD_SIZE` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.gt, 0),
    )
    rank: int = Field(
        default=None,
        desc="Rank of the local process. Typically provided by torchrun or equivalent through the `RANK` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.geq, 0),
    )
    backend: str = Field(
        default="gloo",
        desc="Distributed backend to use.",
        hint=FieldHint.optional,
    )

    def _validate(self) -> None:
        if self.world_size is None:
            self.world_size = self.default_world_size
        if self.rank is None:
            self.rank = self.default_rank
        super()._validate()
        Assert.in_range(self.rank, 0, self.world_size)