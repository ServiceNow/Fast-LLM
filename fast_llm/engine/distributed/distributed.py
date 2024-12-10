import datetime
import logging

import torch
import torch.distributed

from fast_llm.engine.distributed.config import (
    MAX_SEED,
    DistributedConfig,
    DistributedDim,
    DistributedDimNames,
    PhaseType,
)
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


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

        self.world_group = self._process_groups[DistributedDimNames.world]
        self.data_group = self._process_groups[DistributedDimNames.data]
        self.pipeline_group = self._process_groups[DistributedDimNames.pipeline]
        self.tensor_group = self._process_groups[DistributedDimNames.tensor]
        self.sequence_data_group = self._process_groups[DistributedDimNames.sequence_data]
        self.batch_data_group = self._process_groups[DistributedDimNames.batch_data]
        self.tensor_and_sequence_data_group = self._process_groups[DistributedDimNames.tensor_and_sequence_data]

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
        TODO v0.3: Move unrelated content elsewhere.
        """
        seed_shift = step * self._config.sample_seed_shift + self._phase_seeds_shifts[phase]
        self.pp_generator.manual_seed((self._pp_seed + seed_shift) % MAX_SEED)
        self.tp_generator.manual_seed((self._tp_seed + seed_shift) % MAX_SEED)

    def __del__(self):
        # Shutdown the process group backend explicitly to prevent a nccl warning.
        # We can't call `destroy_process_group` directly because pytorch doesn't know about it.
        for group in self._process_groups.values():
            if group is not None and hasattr(group, "_shutdown"):
                group._shutdown()  # noqa
