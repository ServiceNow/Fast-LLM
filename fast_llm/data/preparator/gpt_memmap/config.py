import os
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.data.config import TokenizerConfig
from fast_llm.data.preparator.config import DatasetPreparatorConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.data.preparator.gpt_memmap.prepare import GPTMemmapDatasetPreparator
MEMMAP_DTYPES = {
    1: DataType.uint8,
    2: DataType.int8,
    3: DataType.int16,
    4: DataType.int32,
    5: DataType.int64,
    6: DataType.float32,
    7: DataType.float64,
    8: DataType.uint16,
}
MEMMAP_DTYPES_INV = {y: x for x, y in MEMMAP_DTYPES.items()}
MEMMAP_INDEX_HEADER = b"MMIDIDX\x00\x00"


@config_class
class GPTHuggingfaceDatasetConfig(Config):
    path: str = Field(
        default=None,
        desc="Name or path of the dataset.",
        hint=FieldHint.core,
    )
    config_name: None | str = Field(
        default=None,
        desc="Specific configuration name for the dataset.",
        hint=FieldHint.optional,
    )
    data_directory: None | str = Field(
        default=None,
        desc="data_dir argument passed to `load_dataset`",
        hint=FieldHint.optional,
    )
    data_files: None | str | list[str] = Field(
        default=None,
        desc="data_files argument passed to `load_dataset`",
        hint=FieldHint.optional,
    )
    split: str = Field(
        default="train",
        desc="Split of the dataset to use.",
        hint=FieldHint.optional,
    )
    field: str = Field(
        default="text",
        desc="Field of the dataset to use.",
        hint=FieldHint.optional,
    )
    loss_masking_spans: None | str = Field(
        default=None, desc="Field containing character spans to mask for loss computation", hint=FieldHint.optional
    )
    chosen_text: None | str = Field(
        default=None, desc="Field containing chosen text for preference optimization", hint=FieldHint.optional
    )
    rejected_text: None | str = Field(
        default=None, desc="Field containing rejected text for preference optimization", hint=FieldHint.optional
    )
    data_type: DataType | None = Field(
        default=None,
        desc="Data type of the dataset field."
        " If not provided, it will be inferred based on the tokenizer vocabulary size.",
        hint=FieldHint.optional,
    )
    trust_remote_code: bool = Field(
        default=False,
        desc="Trust remote code when downloading the dataset.",
        hint=FieldHint.optional,
    )
    disable_disk_space_check: bool = Field(
        default=False,
        desc="Disable disk space check. Useful for environments where disk space is not accurately reported.",
        hint=FieldHint.optional,
    )


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


@config_class()
class GPTMemmapDatasetPreparatorConfig(DatasetPreparatorConfig):
    preparator_name: typing.ClassVar[str] = "gpt_memmap"

    output_path: pathlib.Path = Field(
        default=None,
        desc="Output directory for the processed dataset.",
        hint=FieldHint.core,
    )
    distributed: DatasetPreparatorDistributedConfig = Field(
        default_factory=DatasetPreparatorDistributedConfig,
        desc="Configuration for distributed processing.",
        hint=FieldHint.feature,
    )
    tokens_per_shard: int = Field(
        default=10**9,
        desc="Approximate number of tokens per shard.",
        hint=FieldHint.feature,
        valid=check_field(Assert.geq, 10**5),
    )
    loading_workers: int = Field(
        default=1,
        desc="Number of workers in load_dataset() call.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 1),
    )
    tokenize_workers: int = Field(
        default=1,
        desc="Number of workers for tokenization.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 1),
    )
    saving_workers: int = Field(
        default=1,
        desc="Number of processes for saving the data.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 1),
    )
    dataset: GPTHuggingfaceDatasetConfig = Field(
        default_factory=GPTHuggingfaceDatasetConfig,
        desc="Configuration for the dataset.",
        hint=FieldHint.feature,
    )
    tokenizer: TokenizerConfig = Field(
        default_factory=TokenizerConfig,
        desc="Configuration for the tokenizer.",
        hint=FieldHint.feature,
    )
    splits: dict[str, float] | None = Field(
        default=None,
        desc="Split the output dataset into multiple ones (ex, train/valid/test) with the specified ratios."
        " Does not shuffle samples.",
        hint=FieldHint.optional,
    )

    def _validate(self) -> None:
        assert self.tokenizer.path is not None
        if self.dataset.data_type is not None:
            Assert.incl(DataType.from_numpy(self.dataset.data_type.numpy), MEMMAP_DTYPES_INV)
        super()._validate()

    @classmethod
    def get_dataset_preparator_class(cls) -> type["GPTMemmapDatasetPreparator"]:
        from fast_llm.data.preparator.gpt_memmap.prepare import GPTMemmapDatasetPreparator

        return GPTMemmapDatasetPreparator
