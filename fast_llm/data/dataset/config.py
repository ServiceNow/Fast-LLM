import dataclasses
import pathlib
import typing

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed


@dataclasses.dataclass(kw_only=True)
class SamplingConfig:
    # TODO: Have a separate configuration (subset?) for `build`?
    num_samples: int
    seed: int
    cache_directory: pathlib.Path | None
    verbose: bool
    distributed: "Distributed"
