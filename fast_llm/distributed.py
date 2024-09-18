import datetime
import enum
import logging
import os

import torch
import torch.distributed

from fast_llm.config import Config, ConfigDictFormat, Field, FieldHint, check_field, config_class
from fast_llm.core.distributed import ProcessGroup
from fast_llm.utils import Assert, div

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def get_float_dtype(value: torch.dtype | str) -> torch.dtype:
    if not isinstance(value, torch.dtype):
        value = value.split("torch.")[-1]
        Assert.incl(value, _DTYPE_MAP)
        value = _DTYPE_MAP[value]
    assert value.is_floating_point
    return value


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
    _group: ProcessGroup | None

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

    def setup(self, group: ProcessGroup | None):
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
    TODO: Move these unrelated variables elsewhere.
    TODO: Avoid hard-coding distributed dims (use derived class?)
    TODO: Separate distributed space from config?
    """

    __argparse_map__ = {
        "training_dtype": {"type": str},
        "optimization_dtype": {"type": str},
    }

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
        default=int(os.environ.get("WORLD_SIZE", 1)),
        desc="Size of the world group, e.e., total number of GPUs. Typically provided by torchrun or equivalent through the `WORLD_SIZE` environment variable.",
        hint=FieldHint.expert,
        valid=check_field(Assert.gt, 0),
    )
    rank: int = Field(
        default=int(os.environ.get("RANK", 0)),
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
        default=int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
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
    # TODO: Rename to compute_dtype (not just for training)
    training_dtype: torch.dtype = Field(
        default=torch.float32,
        desc="The data type used for the forward and backward passes.",
        hint=FieldHint.core,
        valid=get_float_dtype,
    )
    optimization_dtype: torch.dtype = Field(
        default=torch.float32,
        desc="The data type used for the optimizer.",
        hint=FieldHint.expert,
        valid=get_float_dtype,
    )
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
    def from_dict(
        cls,
        arg_dict: dict,
        format_: ConfigDictFormat = ConfigDictFormat.flat,
        strict: bool = True,
        strict_cls: bool = False,
    ):
        # Backward compatibility
        # Keep it if not strict in case the dict is used later for the model.
        # TODO: Improve.
        if "sequence_first" in arg_dict and strict:
            del arg_dict["sequence_first"]
        if "separate_init_generators" in arg_dict and strict:
            del arg_dict["separate_init_generators"]
        return super().from_dict(arg_dict, format_=format_, strict=strict, strict_cls=strict_cls)


class Distributed:
    """
    A distributed instance holding pointers to the various process groups.
    Also handle global random seeds and generators.

    TODO: Move unrelated content elsewhere.
    TODO: Make more variables private.
    TODO: Clarify cpu support.
    """

    def __init__(self, config: DistributedConfig, use_cpu: bool = False):
        self._config = config.validate()
        self._use_cpu = use_cpu

        if self._use_cpu:
            Assert.eq(self._config.world_size, 1)
            self.device = torch.device("cpu")
        else:
            Assert.in_range_incl(self._config.local_world_size, 1, torch.cuda.device_count())
            torch.cuda.init()
            self.device = torch.device(self._config.local_rank)
            torch.cuda.set_device(self.device)

        timeout = datetime.timedelta(seconds=self._config.distributed_timeout)

        # We bypass `torch.distributed.init_process_group` which makes things way more complicated for no reason.

        # TODO: Allow other init methods?
        # TODO: Allow different timeout for the store?
        if self._config.world_size > 1:
            self._config.log_first_rank("Initializing TCP store.")
            self.store, _, _ = next(
                torch.distributed.rendezvous("env://", self._config.rank, self._config.world_size, timeout=timeout)
            )
        self._process_groups = {}
        for name, distributed_dim in self._config.distributed_dims.items():
            Assert.eq(distributed_dim.name, name)
            self.add_group(distributed_dim)

        self.world_group = self._process_groups["world"]
        self.data_group = self._process_groups["data"]
        self.pipeline_group = self._process_groups["pipeline"]
        self.tensor_group = self._process_groups["tensor"]
        self.sequence_data_group = self._process_groups["sequence_data"]
        self.batch_data_group = self._process_groups["batch_data"]
        self._config.log_first_rank(f"Setting random seeds...")

        dp_shift = self._config.dp_seed_shift * self._config.data_rank
        pp_shift = self._config.pp_seed_shift * self._config.pipeline_rank
        tp_shift = self._config.tp_seed_shift * self._config.tensor_rank

        pp_base_seed = self._config.seed + dp_shift + pp_shift
        tp_base_seed = pp_base_seed + tp_shift
        pp_init_seed = (
            self._config.seed if self._config.reproducible_init else pp_base_seed
        ) + self._config.pp_gen_init_seed_shift
        tp_init_seed = (
            self._config.seed if self._config.reproducible_init else tp_base_seed
        ) + self._config.tp_gen_init_seed_shift

        # Default cpu generator (Only needed to match Megatron initialization.)
        self.default_cpu_generator = torch.random.default_generator
        self.default_cpu_generator.manual_seed(pp_init_seed % MAX_SEED)

        self.pp_generator = torch.Generator(device=self.device)
        self.tp_generator = torch.Generator(device=self.device)
        self.pp_init_generator = torch.Generator(device=self.device)
        self.tp_init_generator = torch.Generator(device=self.device)

        self._pp_seed = (pp_base_seed + self._config.pp_gen_seed_shift) % MAX_SEED
        self._tp_seed = (tp_base_seed + self._config.tp_gen_seed_shift) % MAX_SEED

        self.pp_init_generator.manual_seed(pp_init_seed % MAX_SEED)
        self.tp_init_generator.manual_seed(tp_init_seed % MAX_SEED)

        self._phase_seeds_shifts = {
            PhaseType.training: self._config.train_seed_shift,
            PhaseType.validation: self._config.valid_seed_shift,
            PhaseType.test: self._config.test_seed_shift,
            PhaseType.inference: self._config.test_seed_shift,
        }

        self.set_step(0, PhaseType.training)

    @property
    def config(self):
        return self._config

    def add_group(self, distributed_dim: DistributedDim):
        """
        Add a process group from its definition.
        The group name (`dim`) must be unique within a distributed instance,

        Note: the group id disambiguate between the different groups with the same name on the cluster.
          (Ex.: there is one data-parallel group for each model-parallel rank.)
          There should be exactly one device for each name, group_id and rank.
        TODO: Make private, create all groups through  distributed dims in config.
        """
        Assert.not_incl(distributed_dim.name, self._process_groups)
        prefix = distributed_dim.name if distributed_dim.id is None else f"{distributed_dim.name}_{distributed_dim.id}"

        if distributed_dim.parent is None:
            parent = None
        else:
            Assert.incl(distributed_dim.parent, self._process_groups)
            parent = self._process_groups[distributed_dim.parent]
        if distributed_dim.size == 1:
            group = None
        elif parent and distributed_dim.size == parent.size():
            Assert.eq(distributed_dim.rank, parent.rank())
            group = parent
        else:
            if parent:
                Assert.lt(distributed_dim.size, parent.size())
                Assert.leq(distributed_dim.rank, parent.rank())
            self._config.log_first_rank(f"Initializing group {distributed_dim.name}, size={distributed_dim.size}...")
            group = torch.distributed.ProcessGroupNCCL(
                torch.distributed.PrefixStore(prefix + "/", self.store),
                distributed_dim.rank,
                distributed_dim.size,
                datetime.timedelta(seconds=self._config.distributed_timeout),
            )
        self._process_groups[distributed_dim.name] = group
        distributed_dim.setup(group)
        return group

    def set_step(self, step: int, phase: PhaseType):
        """
        Reseed pytorch for a given training step.
        TODO: Move unrelated content elsewhere.
        """
        seed_shift = step * self._config.sample_seed_shift + self._phase_seeds_shifts[phase]
        self.pp_generator.manual_seed((self._pp_seed + seed_shift) % MAX_SEED)
        self.tp_generator.manual_seed((self._tp_seed + seed_shift) % MAX_SEED)
