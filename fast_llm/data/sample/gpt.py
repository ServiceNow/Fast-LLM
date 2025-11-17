import dataclasses
import typing

from fast_llm.data.sample.abstract import Batch, Sample

if typing.TYPE_CHECKING:
    import torch


@dataclasses.dataclass
class GPTSample(Sample):
    token_ids: "torch.Tensor"
    loss_masking_spans: "torch.Tensor | None" = None
    chosen_span: "torch.Tensor | None" = None
    rejected_span: "torch.Tensor | None" = None
    sequence_lengths: "torch.Tensor | None" = None


@dataclasses.dataclass
class GPTBatch(Batch):
    token_ids: "torch.Tensor"
    loss_masking_spans: "list[torch.Tensor] | None" = None
    sequence_lengths: "list[torch.Tensor] | None" = None
    chosen_spans: "list[torch.Tensor] | None" = None
    rejected_spans: "list[torch.Tensor] | None" = None
