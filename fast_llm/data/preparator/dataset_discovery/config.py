import pathlib
import typing

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.data.preparator.config import DatasetPreparatorConfig
from fast_llm.engine.config_utils.runnable import RunnableConfig

if typing.TYPE_CHECKING:
    from fast_llm.data.preparator.dataset_discovery.prepare import DatasetDiscoveryPreparator


@config_class(dynamic_type={RunnableConfig: "prepare_dataset_discovery", DatasetPreparatorConfig: "dataset_discovery"})
class DatasetDiscoveryConfig(DatasetPreparatorConfig):
    """
    Configuration for the dataset discovery preparator.

    This preparator recursively discovers datasets in a directory and generates
    a blended dataset config with weights proportional to token counts.
    """

    directory: pathlib.Path = Field(
        desc="Directory to search for datasets recursively",
        hint=FieldHint.core,
    )
    output: pathlib.Path = Field(
        desc="Output path for the generated config YAML file",
        hint=FieldHint.core,
    )
    use_file_refs: bool = Field(
        default=True,
        desc="Use file references (type: file) instead of inlining configs",
        hint=FieldHint.optional,
    )
    ignore_paths: list[pathlib.Path] = Field(
        default_factory=list,
        desc="List of paths to ignore during dataset discovery (can be absolute or relative to directory)",
        hint=FieldHint.optional,
    )

    def _validate(self) -> None:
        super()._validate()
        if not self.directory.exists():
            raise ValueError(f"Directory does not exist: {self.directory}")
        if not self.directory.is_dir():
            raise ValueError(f"Path is not a directory: {self.directory}")

    @classmethod
    def get_dataset_preparator_class(cls) -> type["DatasetDiscoveryPreparator"]:
        from fast_llm.data.preparator.dataset_discovery.prepare import DatasetDiscoveryPreparator

        return DatasetDiscoveryPreparator
