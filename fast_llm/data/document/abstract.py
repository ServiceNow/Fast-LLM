import abc
import dataclasses
import typing

from fast_llm.engine.distributed.config import PhaseType
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.engine.distributed.distributed import Distributed
    from fast_llm.tensor import TensorMeta


@dataclasses.dataclass(kw_only=True)
class Document(abc.ABC):
    def to_device_(self, device: "torch.device") -> typing.Self:
        import torch

        for field in dataclasses.fields(self):
            if isinstance(value := getattr(self, field.name), torch.Tensor):
                setattr(self, field.name, value.to(device))
            elif isinstance(value, Document):
                value.to_device_(device)

        return self


@dataclasses.dataclass(kw_only=True)
class ModelInput(Document):
    phase: PhaseType = None
    # A set of intermediate the model should store in `hidden_states` for downstream usage,
    # referred by name or regex pattern.
    # Tensor names are generally of the form `{module_name}.{tensor_name}`.
    # This field is typically populated downstream, depending on the task.
    output_hidden_states: set[str] = dataclasses.field(default_factory=set)
    # The model will populate this with the hidden states specified by `output_hidden_states`,
    # together with the metadata necessary to reconstruct the global tensor.
    hidden_states: "dict[str, tuple[TensorMeta, torch.Tensor]]" = dataclasses.field(default_factory=dict)
    # Cached intermediate states (ex. key and value tensors) from earlier in the sequence.
    # Cached intermediate states (ex. key and value tensors) from earlier in the sequence.
    pasts: list[typing.Any] | None = None
    # If defined, the model will store intermediate states for downstream computation. Used together with `pasts`.
    presents: list[typing.Any] | None = None

    def set_parent_attributes(self, parent: "ModelInput") -> None:
        self.phase = parent.phase
        self.output_hidden_states = parent.output_hidden_states
        self.hidden_states = parent.hidden_states
        self.pasts = parent.pasts
        self.presents = parent.presents

    def to_kwargs(self) -> dict[str, typing.Any]:
        return {
            LanguageModelKwargs.phase: self.phase,
            LanguageModelKwargs.output_hidden_states: self.output_hidden_states,
            LanguageModelKwargs.hidden_states: self.hidden_states,
            AttentionKwargs.past_key_values: self.pasts,
            AttentionKwargs.presents: self.presents,
        }

    @classmethod
    def share_batch_data(cls, model_inputs: "list[ModelInput]", distributed: "Distributed"):
        """
        Gather values depending on the entire data-parallel batch, ex. the total number of labels or documents.
        Should be called in the main process because distributed operations are not available during preprocessing.
        Implemented as a class method so quantities shared by all models inputs are only computed once.
        Note: this may be called more than once (ex. reference model preprocessing), so the method should be idempotent.
        TODO: ====== Use as entry point for batch broadcasting? ======
        """


@dataclasses.dataclass(kw_only=True)
class Batch(Document):
    pass
