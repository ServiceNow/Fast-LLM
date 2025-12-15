import functools
import os
import pathlib
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.data.preparator.config import DatasetPreparatorConfig
from fast_llm.data.preprocessing.image_patch import ImagePatchConfig
from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.config_utils.runnable import RunnableConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.data.preparator.gpt_memmap.prepare import GPTMemmapDatasetPreparator


@config_class(registry=True)
class LanguageModelSourceConfig(Config):
    """
    A schema holding the name of each relevant column in the dataset.
    Setting optional entries will enable the associated feature.

    This is the base class for source schemas. Use `type: text` (default) for
    plain text datasets, or `type: conversation` for chat/conversation datasets.
    """

    text: str = Field(
        default="text",
        desc="Field of the dataset to use.",
        hint=FieldHint.optional,
    )
    loss_masking_spans: None | str = Field(
        default=None, desc="Field containing character spans to mask for loss computation", hint=FieldHint.optional
    )
    chosen_span: None | str = Field(
        default=None, desc="Field containing chosen text for preference optimization", hint=FieldHint.optional
    )
    rejected_span: None | str = Field(
        default=None, desc="Field containing rejected text for preference optimization", hint=FieldHint.optional
    )
    images: None | str = Field(default=None, desc="Field containing images", hint=FieldHint.optional)
    image_positions: None | str = Field(
        default=None, desc="Field containing image positions in the text.", hint=FieldHint.optional
    )

    @functools.cached_property
    def columns(self) -> list[str]:
        columns = [self.text]
        if self.has_loss_masking_span:
            columns.append(self.loss_masking_spans)
        if self.has_preference_spans:
            columns.extend([self.chosen_span, self.rejected_span])
        if self.has_images:
            columns.extend([self.images, self.image_positions])
        return columns

    @functools.cached_property
    def has_loss_masking_span(self) -> bool:
        return self.loss_masking_spans is not None

    @functools.cached_property
    def has_preference_spans(self) -> bool:
        Assert.eq(self.chosen_span is None, self.rejected_span is None)
        return self.chosen_span is not None

    @functools.cached_property
    def has_images(self) -> bool:
        Assert.eq(self.images is None, self.image_positions is None)
        return self.images is not None

    @functools.cached_property
    def has_conversation(self) -> bool:
        """Whether this is a conversation source schema."""
        return False

    def _validate(self):
        super()._validate()
        if self.has_preference_spans and self.has_loss_masking_span:
            raise ValueError(f"Can not enable both loss masking and preference spans.")


@config_class(dynamic_type={LanguageModelSourceConfig: "text"})
class TextSourceConfig(LanguageModelSourceConfig):
    """
    Source schema for plain text datasets (default).

    The dataset should have a text column containing the document text.
    Optionally, it can have additional columns for loss masking spans,
    preference spans (for DPO), or images.
    """

    pass


@config_class(dynamic_type={LanguageModelSourceConfig: "conversation"})
class ConversationSourceConfig(LanguageModelSourceConfig):
    """
    Source schema for chat/conversation datasets (e.g., Tulu 3, ShareGPT, OpenAI format).

    The dataset should have a messages column containing a list of message dicts,
    where each message has 'role' and 'content' keys. Common roles include:
    - 'system': System prompt
    - 'user': User input
    - 'assistant': Model response (trained on by default)
    - 'tool': Tool/function results
    - 'ipython': Code execution results

    The conversation is formatted using the tokenizer's chat template, which must
    contain {% generation %}...{% endgeneration %} markers to define which content
    to train on. Loss masking spans are automatically computed from these markers.

    Example dataset format:
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
    """

    # Override text field - not used directly for conversation format
    text: None | str = Field(
        default=None,
        desc="Not used for conversation format. Text is generated from messages.",
        hint=FieldHint.optional,
    )

    # Conversation-specific fields
    messages: str = Field(
        default="messages",
        desc="Field containing the conversation messages list. Each message should have 'role' and 'content' keys.",
        hint=FieldHint.core,
    )

    add_generation_prompt: bool = Field(
        default=False,
        desc="Whether to add a generation prompt at the end of the conversation. "
        "Typically False for training data.",
        hint=FieldHint.optional,
    )

    @functools.cached_property
    def columns(self) -> list[str]:
        # For conversation format, we read the messages column, not text
        columns = [self.messages]
        # Images can still be used with conversation format
        if self.has_images:
            columns.extend([self.images, self.image_positions])
        return columns

    @functools.cached_property
    def has_conversation(self) -> bool:
        return True

    @functools.cached_property
    def has_loss_masking_span(self) -> bool:
        # Conversation format always generates loss masking spans
        return True

    def _validate(self):
        # Skip parent validation that checks text field
        Config._validate(self)
        if self.has_preference_spans:
            raise ValueError("Preference spans are not supported with conversation format.")
        if self.has_images:
            # Images with conversation format would require computing image positions in the
            # chat-template-formatted text, which is complex and format-dependent.
            # For VLM training with conversations, preprocess the data to plain text format first.
            raise ValueError(
                "Images are not yet supported with conversation format. "
                "For multimodal conversation data, preprocess to plain text format with image positions."
            )


@config_class()
class GPTHuggingfaceDatasetConfig(Config):
    path: str | pathlib.Path = Field(
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
    source_schema: LanguageModelSourceConfig = Field(
        desc="Configuration for the data source.",
        hint=FieldHint.optional,
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
    load_from_disk: bool = Field(
        default=False,
        desc="Use the `load_from_disk` method for datasets saved with `save_to_disk`.",
        hint=FieldHint.feature,
    )


@config_class()
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
    timeout: int = Field(
        default=3600,
        desc="Timeout in seconds for torch distributed operations. Default is 3600.",
        hint=FieldHint.optional,
        valid=check_field(Assert.gt, 0),
    )

    def _validate(self) -> None:
        if self.world_size is None:
            self.world_size = self.default_world_size
        if self.rank is None:
            self.rank = self.default_rank
        super()._validate()
        Assert.in_range(self.rank, 0, self.world_size)


@config_class(dynamic_type={RunnableConfig: "prepare_gpt_memmap", DatasetPreparatorConfig: "gpt_memmap"})
class GPTMemmapDatasetPreparatorConfig(DatasetPreparatorConfig):
    output_path: pathlib.Path = Field(
        default=None,
        desc="Output directory for the processed dataset.",
        hint=FieldHint.core,
    )
    distributed: DatasetPreparatorDistributedConfig = Field(
        desc="Configuration for distributed processing.",
        hint=FieldHint.feature,
    )
    documents_per_shard: int = Field(
        default=10**6,
        desc="Target number of documents per shard.",
        hint=FieldHint.feature,
    )
    num_workers: int = Field(
        default=1,
        desc="Number of parallel workers.",
        hint=FieldHint.optional,
        valid=check_field(Assert.geq, 1),
    )
    dataset: GPTHuggingfaceDatasetConfig = Field(
        desc="Configuration for the dataset.",
        hint=FieldHint.feature,
    )
    tokenizer: TokenizerConfig = Field(
        desc="Configuration for the tokenizer.",
        hint=FieldHint.feature,
    )
    image_patches: ImagePatchConfig = Field(
        desc="Configuration for the image patches, if enabled.",
        hint=FieldHint.feature,
    )
    splits: dict[str, float] | None = Field(
        default=None,
        desc="Split the output dataset into multiple ones (ex, train/valid/test) with the specified ratios."
        " Does not shuffle samples.",
        hint=FieldHint.optional,
    )

    def _validate(self) -> None:
        super()._validate()
        assert self.tokenizer.path is not None

    @classmethod
    def get_dataset_preparator_class(cls) -> type["GPTMemmapDatasetPreparator"]:
        from fast_llm.data.preparator.gpt_memmap.prepare import GPTMemmapDatasetPreparator

        return GPTMemmapDatasetPreparator
