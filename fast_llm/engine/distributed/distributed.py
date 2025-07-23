import datetime
import logging
import typing

import torch
import torch.distributed

from fast_llm.config import Configurable
from fast_llm.core.distributed import ProcessGroup
from fast_llm.engine.distributed.config import (
    MAX_SEED,
    DistributedConfig,
    DistributedDim,
    DistributedDimNames,
    PhaseType,
    check_ranks_in_range,
)
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class ProcessGroupPool:
    def __init__(
        self,
        rank: int | None = None,
        world_size: int | None = None,
        local_world_size: int | None = None,
        timeout: float = 60,
        use_cpu: bool = False,
    ):

        self._rank = DistributedConfig.default_rank if rank is None else rank
        self._world_size = DistributedConfig.default_world_size if world_size is None else world_size
        self._local_world_size = (
            DistributedConfig.default_local_world_size if local_world_size is None else local_world_size
        )
        self._timeout = timeout
        self._use_cpu = use_cpu
        self._process_groups = {}

        if self._use_cpu:
            Assert.eq(self._world_size, 1)
            self._device = torch.device("cpu")
        else:
            Assert.in_range_incl(self._local_world_size, 1, torch.cuda.device_count())
            torch.cuda.init()
            self._device = torch.device(self._rank % self._local_world_size)
            torch.cuda.set_device(self._device)

        if self._world_size > 1:
            if self._rank == 0:
                logger.info("Initializing TCP store.")
            # We bypass `torch.distributed.init_process_group` which makes things way more complicated for no reason.
            # TODO: Allow other init methods?
            self.store, _, _ = next(
                torch.distributed.rendezvous(
                    "env://",
                    self._rank,
                    self._world_size,
                    timeout=datetime.timedelta(seconds=timeout),
                )
            )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def local_world_size(self):
        return self._local_world_size

    @property
    def device(self):
        return self._device

    def get_process_group(self, global_ranks: range | tuple, group_rank: int) -> ProcessGroup | None:
        """
        Get the requested process group from the pool, or create it if it doesn't exist.
        """
        group_size = len(global_ranks)
        Assert.eq(global_ranks[group_rank], self._rank)
        check_ranks_in_range(global_ranks, 0, self._world_size)
        if group_size == 1:
            return None

        for group_ranks, group in self._process_groups.items():
            # Check if an equivalent group already exists.
            if type(group_ranks) != type(global_ranks):
                if group_ranks == global_ranks:
                    return group
            elif tuple(group_ranks) == tuple(global_ranks):
                return group

        prefix = (
            f"range_{global_ranks.start}_{global_ranks.stop}_{global_ranks.step}"
            if isinstance(global_ranks, range)
            else f"ranks_{"_".join(str(rank) for rank in global_ranks)}"
        )

        group = torch.distributed.ProcessGroupNCCL(
            torch.distributed.PrefixStore(prefix + "/", self.store),
            group_rank,
            group_size,
            datetime.timedelta(seconds=self._timeout),
        )
        self._process_groups[global_ranks] = group
        return group

    def __enter__(self):
        global _default_pool
        assert _default_pool is None
        _default_pool = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _default_pool
        assert _default_pool is self
        _default_pool = None
        self.shutdown()

    def shutdown(self):
        # Shutdown the process group backend explicitly to prevent a nccl warning.
        # We can't call `destroy_process_group` directly because pytorch doesn't know about it.
        for group in self._process_groups.values():
            group.shutdown()

    def __del__(self):
        self.shutdown()


_default_pool: ProcessGroupPool | None = None


class Distributed[ConfigType: DistributedConfig](Configurable[ConfigType]):
    """
    A distributed instance holding pointers to the various process groups.
    Also handle global random seeds and generators.

    TODO: Move unrelated content elsewhere.
    TODO: Make more variables private.
    TODO: Clarify cpu support.
    """

    config_class: typing.ClassVar[type[DistributedConfig]] = DistributedConfig

    def __init__(self, config: DistributedConfig, use_cpu: bool = False):
        super().__init__(config)
        assert self._config.reference_config is None

        self._local_pool = _default_pool is None
        if self._local_pool:
            self._pool = ProcessGroupPool(
                self._config.rank,
                self._config.world_size,
                self._config.local_world_size,
                self._config.timeout,
                use_cpu,
            )
        else:
            self._pool = _default_pool
            Assert.geq(self._pool.world_size, self._config.world_size)
            Assert.eq(self._pool.rank, self._config.rank)
            Assert.geq(self._pool.local_world_size, self._config.local_world_size)
            Assert.eq(self._pool.device.type, "cpu" if use_cpu else "cuda")

        self.world_group = self.add_group(self._config.distributed_dims[DistributedDimNames.world])
        self.data_group = self.add_group(self._config.distributed_dims[DistributedDimNames.data])
        self.pipeline_group = self.add_group(self._config.distributed_dims[DistributedDimNames.pipeline])
        self.tensor_group = self.add_group(self._config.distributed_dims[DistributedDimNames.tensor])
        self.sequence_data_group = self.add_group(self._config.distributed_dims[DistributedDimNames.sequence_data])
        self.batch_data_group = self.add_group(self._config.distributed_dims[DistributedDimNames.batch_data])
        self.tensor_and_sequence_data_group = self.add_group(
            self._config.distributed_dims[DistributedDimNames.tensor_and_sequence_data]
        )

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
    def device(self):
        return self._pool.device

    def add_group(self, distributed_dim: DistributedDim) -> ProcessGroup | None:
        """
        Add a process group from its definition.
        """
        self._config.log_first_rank(f"Initializing group {distributed_dim.name}, size={distributed_dim.size}...")
        distributed_dim.check_ranks_in_range(0, self._config.world_size)
        group = self._pool.get_process_group(distributed_dim.global_ranks, distributed_dim.rank)
        distributed_dim.setup(group)
        return group

    def check_config(self, config: DistributedConfig) -> None:
        # Allows using this `Distributed` on a model with a distributed config that is a copy of `self._config`
        if config.reference_config is None:
            Assert.is_(config, self._config)
        else:
            Assert.is_(config.reference_config, self._config)

    def set_step(self, step: int, phase: PhaseType) -> None:
        """
        Reseed pytorch for a given training step.
        TODO v0.3: Move unrelated content elsewhere.
        """
        seed_shift = step * self._config.sample_seed_shift + self._phase_seeds_shifts[phase]
        self.pp_generator.manual_seed((self._pp_seed + seed_shift) % MAX_SEED)
        self.tp_generator.manual_seed((self._tp_seed + seed_shift) % MAX_SEED)

    def __del__(self):
        if self._local_pool:
            self._pool.shutdown()
