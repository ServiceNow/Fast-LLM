import dataclasses

from fast_llm.engine.config_utils.data_type import DataType

# TODO: Store span type?
# class SpanType(enum.StrEnum):
#    none = "none"
#    loss_masking = "loss_masking"
#    preference = "preference"


@dataclasses.dataclass(kw_only=True)
class GPTMemmapDatasetHeader:
    num_documents: int
    token_data_type: DataType = DataType.int64
    has_spans: bool = False
    has_images: bool = False

    def __post_init__(self):
        self.token_data_type = DataType(self.token_data_type)
