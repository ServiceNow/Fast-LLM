import typing

from fast_llm.config import NoAutoValidate
from fast_llm.engine.base_model.base_model import Layer, LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.audio_encoder.adapter import AudioAdapter
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioKwargs
from fast_llm.layers.audio_encoder.encoder import AudioConv
from fast_llm.layers.block.block import BlockBase
from fast_llm.layers.common.peft.config import PeftConfig

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed


class AudioEncoder[ConfigType: AudioEncoderConfig](BlockBase[ConfigType]):
    """
    Full audio encoder stack: AudioConv → Transformer blocks → AudioAdapter.

    Analogous to ``layers/vision/vision_encoder.py::VisionEncoder``.
    The ``hidden_dim`` passed in from the parent model is the LM hidden size;
    the internal audio encoder hidden size is ``config.hidden_size``.
    """

    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(config, distributed_config, hidden_dim=hidden_dim, lr_scale=lr_scale, peft=peft)

        # Internal hidden dim for the audio encoder (may differ from LM hidden_dim)
        self._audio_hidden_dim = TensorDim("audio_hidden", self._config.hidden_size)

        self.conv = AudioConv(self._config, distributed_config)

        # The audio transformer blocks must NOT use sequence tensor parallelism.
        # The audio sequence is always replicated across all TP ranks (never split), so
        # the STP gather_op inside each attention projection would create TP copies of
        # the same token sequence.  Flash attention then only processes the first
        # N_total tokens (from cu_seqlens), leaving garbage in the remaining 7/8 of
        # the tensor.  After the dense reduce-scatter the values are amplified by TP
        # at every layer, quickly overflowing bf16 to NaN.
        #
        # Fix: create an unvalidated copy of distributed_config with STP disabled,
        # then set its reference_config back to the original so that
        # distributed.check_config() passes (it checks reference_config identity) and
        # distributed_dims delegates to the original (sharing the same process groups).
        # This follows the same pattern used in training/config.py for reference models.
        if distributed_config.sequence_tensor_parallel:
            with NoAutoValidate():
                audio_distributed_config = distributed_config.to_copy(
                    {"sequence_tensor_parallel": False}
                )
            audio_distributed_config.reference_config = distributed_config
        else:
            audio_distributed_config = distributed_config

        self.encoder = self._config.encoder.get_layer(
            audio_distributed_config,
            self._audio_hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        # AudioAdapter projects from audio hidden → LM hidden (self._hidden_dim)
        self.adapter = AudioAdapter(
            self._config,
            audio_hidden_dim=self._audio_hidden_dim,
            output_dim=self._hidden_dim,
            distributed_config=distributed_config,
        )

    def setup(self, distributed: "Distributed") -> None:
        # AudioEncoder exposes its sub-layers (conv, encoder blocks, adapter) via
        # get_layers(), so LayerBase.setup() on the parent model sets them up through
        # the LayerWithNamespace traversal.  AudioEncoder itself is never in get_layers(),
        # so we only record _distributed here (needed by preprocess()); sub-layer setup
        # is intentionally left to the parent traversal.
        distributed.check_config(self._distributed_config)
        self._distributed = distributed
        self._is_setup = True

    def get_layers(self) -> list[Layer]:
        return [self.conv] + self.encoder.get_layers() + [self.adapter]

    def get_preprocessing_config(self) -> dict[str, typing.Any]:
        return self.encoder.get_preprocessing_config()

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        if self._config.enabled and AudioKwargs.audio in kwargs:
            # Run mel-spectrogram extraction and set audio-specific attention kwargs.
            if not hasattr(self, "_audio_preprocessor"):
                from fast_llm.layers.audio_encoder.preprocessing import AudioPreprocessor

                self._audio_preprocessor = AudioPreprocessor(
                    self._config,
                    device=self._distributed.device,
                )
            self._audio_preprocessor.preprocess(None, kwargs)
        self.encoder.preprocess(kwargs)

    def get_loss_definitions(self) -> list[LossDef]:
        return self.encoder.get_loss_definitions()
