import typing

import torch

from fast_llm.config import NoAutoValidate
from fast_llm.engine.base_model.base_model import Layer, LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDim, DistributedDimNames
from fast_llm.layers.audio_encoder.adapter import AudioAdapter
from fast_llm.layers.audio_encoder.config import AudioEncoderConfig, AudioKwargs
from fast_llm.layers.audio_encoder.encoder import AudioConv
from fast_llm.layers.block.block import BlockBase
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.tensor import TensorMeta

if typing.TYPE_CHECKING:
    from fast_llm.engine.distributed.distributed import Distributed


class _AudioFinalNorm(Layer):
    """
    Wraps the audio encoder's final normalization in a Layer so it slots into
    AudioEncoder.get_layers() between the transformer blocks and the adapter.

    For Whisper-family checkpoints this receives ``encoder.layer_norm``
    (a LayerNorm applied after the last encoder block). It used to live in
    ``AudioAdapter.norm_1``; moving it here frees ``adapter.norm_1`` to act as
    the projector's own pre-norm (e.g. Ultravox's ``ln_pre``).

    The norm shares the parent ``AudioEncoder``'s ``lr_scale`` and ``peft``,
    matching the convention used by intra-block norms (see
    ``DecoderBlock.norm_1``). Freezing the audio encoder via
    ``audio_encoder.lr_scale: 0.0`` therefore also freezes this norm and frees
    autograd from retaining its ``(N_clips × T × hidden)`` input activation.
    """

    def __init__(
        self,
        config: AudioEncoderConfig,
        audio_hidden_dim: TensorDim,
        distributed_config: DistributedConfig,
        *,
        lr_scale: float | None = None,
        peft: PeftConfig | None = None,
    ):
        super().__init__(distributed_config)
        self._audio_hidden_dim = audio_hidden_dim
        self.norm = config.normalization.get_layer(audio_hidden_dim, lr_scale=lr_scale, peft=peft)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            return input_
        return self.norm(input_)


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
        #
        # Additionally, when ``disable_tensor_parallel`` is set, we replace the
        # ``tensor`` entry of distributed_dims with a size-1 DistributedDim so
        # head-sharding is bypassed (required when the encoder's attention head
        # count does not divide the LM's tensor_parallel — e.g. Whisper-large-v3-turbo
        # has 20 heads, which fails on TP=8). The encoder weights are then
        # replicated across the TP group; collectives no-op on the size-1 group.
        needs_audio_config_override = (
            distributed_config.sequence_tensor_parallel
            or (self._config.disable_tensor_parallel and distributed_config.tensor_parallel > 1)
        )
        if needs_audio_config_override:
            with NoAutoValidate():
                audio_distributed_config = distributed_config.to_copy(
                    {
                        "tensor_parallel": (
                            1 if self._config.disable_tensor_parallel else distributed_config.tensor_parallel
                        ),
                        "sequence_tensor_parallel": False,
                    }
                )
            audio_distributed_config.reference_config = distributed_config
            if self._config.disable_tensor_parallel and distributed_config.tensor_parallel > 1:
                # Override the cached ``tensor`` entry of distributed_dims with a size-1
                # dim. ``check_config`` still passes via reference_config identity; head
                # sharding sees parallel_dim.size=1 and TP collectives become no-ops.
                audio_tensor_dim = DistributedDim(
                    name=DistributedDimNames.tensor,
                    size=1,
                    rank=0,
                    global_ranks=range(distributed_config.rank, distributed_config.rank + 1),
                )
                audio_tensor_dim.setup(None)
                audio_distributed_config.__dict__["distributed_dims"] = {
                    **distributed_config.distributed_dims,
                    DistributedDimNames.tensor: audio_tensor_dim,
                }
        else:
            audio_distributed_config = distributed_config

        self.encoder = self._config.encoder.get_layer(
            audio_distributed_config,
            self._audio_hidden_dim,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

        # Final norm applied after the last transformer block, before the adapter.
        # Receives the HF Whisper-family ``encoder.layer_norm`` weights.
        # Inherits ``lr_scale`` / ``peft`` from the AudioEncoder so that freezing
        # the audio encoder via YAML also freezes this norm — matching the
        # in-block convention (e.g. ``DecoderBlock.norm_1``).
        self.final_norm = _AudioFinalNorm(
            self._config,
            audio_hidden_dim=self._audio_hidden_dim,
            distributed_config=distributed_config,
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
        return [self.conv] + self.encoder.get_layers() + [self.final_norm, self.adapter]

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
