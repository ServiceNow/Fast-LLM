import enum
import logging
import os
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    from fast_llm.core.distributed import ProcessGroup


logger = logging.getLogger(__name__)


_BIG_PRIMES = (
    317767863445754100075399033823,
    811026271858507938190098775099,
    500460795349110443334056239993,
    631112027069964424381239824623,
    705275193289568515128435800471,
    278779420836085904093221202933,
    894750739684993243926471979237,
    751127116949963770353413160199,
    938219878163699459065752841447,
    683552447587140661489672773353,
    220111337975202516901860145957,
    501974169931277706872159392843,
    968032476254500041143685844117,
    825686112544887677065408128553,
    268764172324680632509040370823,
    900176227183325302080254561171,
    335974126039268762110607666959,
    107500537994172876430818236981,
    194940155773131779019247515701,
    448005733546283060155968310919,
)
MAX_SEED = 2**64


class PhaseType(str, enum.Enum):
    training = "Training"
    validation = "Validation"
    test = "Test"
    inference = "Inference"

    @property
    def is_training(self) -> bool:
        return self == PhaseType.training


class DistributedDim:
    """
    A dataclass to hold all relevant information on a process group without actually creating it.
    """

    _is_setup: bool = False
    _group: typing.Optional["ProcessGroup"]

    def __init__(self, name: str, size: int = 1, rank: int = 0, id_: str | None = None, parent: str | None = None):
        self._name = name
        self._size = size
        self._rank = rank
        self._id = id_
        self._parent = parent

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return self._rank

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        return self._parent

    @property
    def group(self):
        assert self._is_setup
        return self._group

    def __repr__(self):
        return (
            f"DistributedDim(name={self.name}, size={self.size}, rank={self.rank}, id={self.id}, parent={self.parent})"
        )

    def setup(self, group: typing.Optional["ProcessGroup"]):
        assert not self._is_setup
        self._is_setup = True
        Assert.eq(group is None, self.size == 1)
        if group is not None:
            Assert.eq(group.size(), self._size)
            Assert.eq(group.rank(), self._rank)
        self._group = group


class DistributedDimNames:
    # A set of common distributed dim names packed into a singleton.
    world = "world"
    tensor = "tensor"
    data = "data"
    pipeline = "pipeline"
    sequence_data = "sequence_data"
    batch_data = "batch_data"


@config_class()
class DistributedConfig(Config):
    """
    Configuration for the distributed setup.
    Also include variables for global settings such as data types, random seeds, initialization parameters.
    TODO v0.2: Move these unrelated variables elsewhere.
    TODO: Avoid hard-coding distributed dims (use derived class?)
    TODO: Separate distributed space from config?
    """

    default_world_size: typing.ClassVar[int] = int(os.environ.get("WORLD_SIZE", 1))
    default_local_world_size: typing.ClassVar[int] = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    default_rank: typing.ClassVar[int] = int(os.environ.get("RANK", 0))

    tensor_parallel: int = Field(
        default=1, desc="Tensor parallelism group size.", hint=FieldHint.performance, valid=check_field(Assert.gt, 0)
    )
    pipeline_parallel: int = Field(
        default=1, desc="Pipeline parallelism group size.", hint=FieldHint.performance, valid=check_field(Assert.gt, 0)
    )
    data_parallel: int = Field(init=False, desc="Data parallelism group size.", hint=FieldHint.derived)
    model_parallel: int = Field(
        init=False, desc="Model parallelism group size (tensor * pipeline).", hint=FieldHint.derived
    )
    num_nodes: int = Field(init=False, desc="Number of GPU nodes.", hint=FieldHint.derived)
    sequence_tensor_parallel: bool = Field(
        default=False, desc="Enable sequence tensor parallelism.", hint=FieldHint.performance
    )
    sequence_data_parallel: int = Field(
        default=1,
        desc="Sequence data parallelism group size.",
        hint=FieldHint.performance,
        valid=check_field(Assert.gt, 0),
    )
    batch_data_parallel: int = Field(init=False, desc="Batch data parallelism group size.", hint=FieldHint.performance)
    world_size: int = Field(
        default=None,
        desc="Size of the world group, e.e., total number of GPUs. Typically provided by torchrun or equivalent through the `WORLD_SIZE` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.gt, 0),
    )
    rank: int = Field(
        default=None,
        desc="Rank of the local process. Typically provided by torchrun or equivalent through the `RANK` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.geq, 0),
    )
    data_rank: int = Field(init=False, desc="Data-parallel rank of the local process.", hint=FieldHint.derived)
    pipeline_rank: int = Field(init=False, desc="Pipeline-parallel rank of the local process.", hint=FieldHint.derived)
    tensor_rank: int = Field(init=False, desc="Tensor-parallel rank of the local process.", hint=FieldHint.derived)
    sequence_data_rank: int = Field(
        init=False, desc="Sequence-data-parallel rank of the local process.", hint=FieldHint.derived
    )
    batch_data_rank: int = Field(
        init=False, desc="Batch-data-parallel rank of the local process.", hint=FieldHint.derived
    )
    distributed_dims: dict[str, DistributedDim] = Field(
        init=False, desc="The `DistributedDim` objects for the distributed dimensions.", hint=FieldHint.derived
    )
    local_rank: int = Field(
        init=False,
        desc="The rank of the process on the current node.",
        hint=FieldHint.derived,
    )
    local_world_size: int = Field(
        default=None,
        desc="Number of GPUs in each node. Typically provided by torchrun or equivalent through the `LOCAL_WORLD_SIZE` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.gt, 0),
    )
    pipeline_first: bool = Field(
        default=False,
        desc="Prioritize the pipeline groups for placement of nearby ranks over data groups.",
        hint=FieldHint.expert,
    )
    distributed_timeout: float = Field(
        default=60,
        desc="Timeout for distributed operations.",
        hint=FieldHint.optional,
        valid=check_field(Assert.gt, 0),
    )
    seed: int = Field(default=1234, desc="A seed for training.", hint=FieldHint.optional)
    # TODO v0.2: Rename to compute_dtype (not just for training), move elsewhere
    training_dtype: DataType = Field(
        default=DataType.float32,
        desc="The data type used for the forward and backward passes.",
        hint=FieldHint.core,
    )
    # TODO v0.2: move elsewhere
    optimization_dtype: DataType = Field(
        default=DataType.float32,
        desc="The data type used for the optimizer.",
        hint=FieldHint.expert,
    )
    # TODO v0.2: move random state elsewhere
    # Extra seed parameters (can usually be left alone)
    dp_seed_shift: int = Field(
        default=_BIG_PRIMES[0], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    pp_seed_shift: int = Field(
        default=_BIG_PRIMES[1], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    pp_gen_seed_shift: int = Field(
        default=_BIG_PRIMES[2], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    pp_gen_init_seed_shift: int = Field(
        default=_BIG_PRIMES[3], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    tp_seed_shift: int = Field(
        default=_BIG_PRIMES[4], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    tp_gen_seed_shift: int = Field(
        default=_BIG_PRIMES[5], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    tp_gen_init_seed_shift: int = Field(
        default=_BIG_PRIMES[6], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    sample_seed_shift: int = Field(
        default=_BIG_PRIMES[7], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    train_seed_shift: int = Field(
        default=_BIG_PRIMES[8], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    valid_seed_shift: int = Field(
        default=_BIG_PRIMES[9], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    test_seed_shift: int = Field(
        default=_BIG_PRIMES[10], desc="Seed shift for extra randomness.", hint=FieldHint.optional
    )
    # (slower, uses more memory, mainly for debug)
    reproducible_init: bool = Field(
        default=False,
        desc="Ensure the initialization is the same for any distributed configuration.",
        hint=FieldHint.testing,
    )

    def _validate(self):
        if self.world_size is None:
            self.world_size = self.default_world_size
        if self.rank is None:
            self.rank = self.default_rank
        if self.local_world_size is None:
            self.local_world_size = self.default_local_world_size
        self.model_parallel = self.tensor_parallel * self.pipeline_parallel
        self.data_parallel = div(self.world_size, self.model_parallel)
        self.num_nodes = div(self.world_size, self.local_world_size)
        self.local_rank = self.rank % self.local_world_size
        Assert.multiple(self.local_world_size, self.tensor_parallel)

        if self.pipeline_first:
            # Smaller models can be more demanding on pipeline parallel.
            self.data_rank = (self.rank // self.tensor_parallel) // self.pipeline_parallel
            self.pipeline_rank = (self.rank // self.tensor_parallel) % self.pipeline_parallel
        else:
            # Larger models are more demanding on data parallel.
            self.data_rank = (self.rank // self.tensor_parallel) % self.data_parallel
            self.pipeline_rank = (self.rank // self.tensor_parallel) // self.data_parallel
        self.sequence_data_rank = self.data_rank % self.sequence_data_parallel
        self.batch_data_parallel = div(self.data_parallel, self.sequence_data_parallel)
        self.batch_data_rank = self.data_rank // self.sequence_data_parallel

        self.tensor_rank = self.rank % self.tensor_parallel
        if self.tensor_parallel == 1:
            self.sequence_tensor_parallel = False

        self.distributed_dims = {}

        self.add_distributed_dim(
            DistributedDim(name=DistributedDimNames.world, size=self.world_size, rank=self.rank, id_=None, parent=None)
        )
        self.add_distributed_dim(
            DistributedDim(
                name=DistributedDimNames.data,
                size=self.data_parallel,
                rank=self.data_rank,
                id_=f"x_{self.pipeline_rank}_{self.tensor_rank}",
                parent=DistributedDimNames.world,
            )
        )
        self.add_distributed_dim(
            DistributedDim(
                name=DistributedDimNames.pipeline,
                size=self.pipeline_parallel,
                rank=self.pipeline_rank,
                id_=f"x_{self.data_rank}_{self.tensor_rank}",
                parent=DistributedDimNames.world,
            )
        )
        self.add_distributed_dim(
            DistributedDim(
                name=DistributedDimNames.tensor,
                size=self.tensor_parallel,
                rank=self.tensor_rank,
                id_=f"x_{self.data_rank}_{self.pipeline_rank}",
                parent=DistributedDimNames.world,
            )
        )
        self.add_distributed_dim(
            DistributedDim(
                name=DistributedDimNames.sequence_data,
                size=self.sequence_data_parallel,
                rank=self.sequence_data_rank,
                id_=f"{self.batch_data_rank}_{self.pipeline_rank}_{self.tensor_rank}",
                parent=DistributedDimNames.data,
            )
        )
        self.add_distributed_dim(
            DistributedDim(
                name=DistributedDimNames.batch_data,
                size=self.batch_data_parallel,
                rank=self.batch_data_rank,
                id_=f"{self.sequence_data_rank}_{self.pipeline_rank}_{self.tensor_rank}",
                parent=DistributedDimNames.data,
            )
        )

        super()._validate()

        Assert.in_range(self.rank, 0, self.world_size)
        Assert.in_range(self.local_rank, 0, self.local_world_size)

    def add_distributed_dim(self, distributed_dim: DistributedDim):
        if distributed_dim.name in self.distributed_dims:
            Assert.eq(distributed_dim, self.distributed_dims[distributed_dim.name])
        else:
            if distributed_dim.parent is not None:
                assert distributed_dim.parent in self.distributed_dims
            self.distributed_dims[distributed_dim.name] = distributed_dim

    def get_distributed_dim(self, name: str):
        return self.distributed_dims[name]

    def _log_on_rank(self, *message, rank: int | None = None, log_fn=logger.info):
        if rank is None or self.rank == rank:
            log_fn(", ".join([str(m) for m in message]))

    def log_first_rank(self, *message, log_fn=logger.info):
        return self._log_on_rank(*message, rank=0, log_fn=log_fn)

    @classmethod
    def _from_dict(
        cls,
        default: dict,
        strict: bool = True,
        flat: bool = False,
    ):
        # TODO v0.2: Remove backward compatibility fix
        if "sequence_first" in default and strict:
            del default["sequence_first"]
        if "separate_init_generators" in default and strict:
            del default["separate_init_generators"]
        return super()._from_dict(default, strict, flat)
