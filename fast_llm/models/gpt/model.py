import logging
import typing

import torch

from fast_llm.data.data.gpt.data import GPTBatch
from fast_llm.engine.base_model.base_model import BaseModel, Layer
from fast_llm.engine.base_model.config import LossDef
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames, PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.config import BlockDimNames
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTBatchConfig, GPTModelConfig
from fast_llm.models.gpt.megatron import get_init_megatron
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class GPTBaseModel[ConfigType: GPTBaseModelConfig](BaseModel[ConfigType]):
    """
    A transformer-based language model generalizing the GPT model architecture.
    """

    _config: ConfigType

    def __init__(
        self,
        config: GPTBaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        self._hidden_dim = TensorDim("hidden", config.embeddings_layer.hidden_size)
        super().__init__(config, distributed_config)

        hidden_dim = TensorDim("hidden", self.embeddings_layer.hidden_size)
        self.embedding = self._config.embeddings_layer.get_layer(
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=None,
            peft=self._config.peft,
        )
        self.decoder = self._config.decoder.get_layer(
            distributed_config,
            hidden_dim,
            lr_scale=None,
            peft=self._config.peft,
        )
        self.head = self._config.output_layer.get_layer(
            distributed_config,
            self._config.embeddings_layer,
            hidden_dim=hidden_dim,
            lr_scale=None,
            peft=self._config.peft,
        )

        if self._config.use_megatron_initialization:
            for param in self.parameters():
                Assert.custom(isinstance, param, ParameterMeta)
                param.init_parameter = get_init_megatron(
                    param, self._config.decoder.block, config.embeddings_layer.hidden_size
                )  # Noqa

        # TODO ====== Vision ======
        # if self._config.vision_encoder.enabled:
        #    self._preprocessors.append(VisionPreprocessor(self._config.vision_encoder, self._tensor_space))
        #    self._preprocessors.append(self._config.vision_encoder.transformer.rotary.build(self._tensor_space))

    def get_layers(self) -> list["Layer"]:
        return self.embedding.get_layers() + self.decoder.get_layers() + self.head.get_layers()

    # TODO ====== Vision ======
    # def get_vision_layers(self) -> list[Layer]:
    #    vit_layers = [
    #        VisionTransformerBlock(self._config.vision_encoder.transformer, self._tensor_space, block_index=idx + 1)
    #        for idx in range(self._config.vision_encoder.transformer.num_layers)
    #    ]
    #    return [
    #        PatchConv(self._config.vision_encoder, self._tensor_space),
    #        *vit_layers,
    #        VisionAdapter(self._config.vision_encoder, self._tensor_space),
    #        MultiModalEmbedding(self._config, self._tensor_space),
    #    ]

    def preprocess_meta(
        self, batch_meta: GPTBatchConfig | torch.Tensor, phase: PhaseType
    ) -> list[tuple[TensorMeta, dict]]:
        # TODO ====== Remove (Move batch splitting elsewhere) ======
        # TODO: Use parallel/sequential dims, distinguish micro and full batch/sequence

        if isinstance(batch_meta, GPTBatchConfig):
            micro_batch_size = batch_meta.micro_batch_size
            sequence_length = batch_meta.sequence_length
            micro_sequence_length = batch_meta.micro_sequence_length
            truncate_documents = batch_meta.truncate_documents
        else:
            micro_batch_size, sequence_length = batch_meta.shape
            if phase != PhaseType.inference:
                sequence_length -= self._config.output_layer.prediction_heads
            micro_sequence_length = sequence_length
            truncate_documents = True

        batch_data = self._distributed_config.get_distributed_dim(DistributedDimNames.batch_data)
        batch_dim = TensorDim(BlockDimNames.batch, micro_batch_size * batch_data.size, batch_data)

        if micro_sequence_length is None:
            micro_sequence_length = sequence_length
        else:
            Assert.multiple(sequence_length, micro_sequence_length)

        # TODO: Calculate hidden dims elsewhere?
        sequence_q_dim = TensorDim(
            BlockDimNames.sequence_q,
            micro_sequence_length,
            self._distributed_config.get_distributed_dim(DistributedDimNames.sequence_data),
        )
        hidden_sequence_q_dim = (
            TensorDim(
                BlockDimNames.sequence_q_tp,
                micro_sequence_length,
                self._distributed_config.get_distributed_dim(DistributedDimNames.tensor_and_sequence_data),
            )
            if self._distributed_config.sequence_tensor_parallel
            else sequence_q_dim
        )

        need_sequence_first = hidden_sequence_q_dim.size != sequence_length
        if self._config.sequence_first is None:
            sequence_first = need_sequence_first
        else:
            sequence_first = self._config.sequence_first
            assert not (need_sequence_first and not sequence_first)

        hidden_dims = (
            (hidden_sequence_q_dim, batch_dim, self._hidden_dim)
            if sequence_first
            else (batch_dim, hidden_sequence_q_dim, self._hidden_dim)
        )

        common_kwargs = {
            LanguageModelKwargs.phase: phase,
            AttentionKwargs.sequence_first: sequence_first,
            AttentionKwargs.hidden_dims: hidden_dims,
            AttentionKwargs.sequence_length: sequence_length,
            AttentionKwargs.sequence_q_dim: sequence_q_dim,
            LanguageModelKwargs.mask_inputs: not truncate_documents,
        }

        # TODO ====== Vision ======
        # if self._config.vision_encoder.enabled:
        #    try:
        #        max_image_size = batch_meta.max_image_size
        #    except AttributeError:
        #        max_image_size = 256
        #        logger.warning("Inference mode: max_image_size not provided, defaulting to 256")
        #    vision_kwargs = {
        #        VisionEncoderKwargs.patch_size: self._config.vision_encoder.patch_size,
        #        VisionEncoderKwargs.max_image_size: max_image_size,
        #        VisionEncoderKwargs.rope_theta: self._config.vision_encoder.transformer.rotary.theta,
        #        VisionEncoderKwargs.kv_channels: self._tensor_space[VisionTransformerDimNames.kv_channels].size,
        #        VisionEncoderKwargs.out_channels: self._tensor_space[VisionEncoderDimNames.out_channels].size,
        #    }
        #    vision_hidden_dim = self._tensor_space[VisionTransformerDimNames.hidden]
        #    vision_hidden_dims = (
        #        (hidden_sequence_q_dim, batch_dim, vision_hidden_dim)
        #        if sequence_first
        #        else (batch_dim, hidden_sequence_q_dim, vision_hidden_dim)
        #    )
        #    vision_kwargs.update(
        #        {
        #            VisionTransformerKwargs.hidden_dims: vision_hidden_dims,
        #        }
        #    )
        #    common_kwargs.update(vision_kwargs)

        sequence_k_pasts = range(
            sequence_q_dim.size * self._distributed_config.sequence_data_rank,
            sequence_length,
            micro_sequence_length,
        )
        reference_preprocessed_metas = {}
        for name, reference_model in self._reference_models.items():
            reference_preprocessed_metas[name] = reference_model.fast_llm_model.base_model.preprocess_meta(
                batch_meta, PhaseType.inference
            )
            Assert.eq(len(reference_preprocessed_metas[name]), len(sequence_k_pasts))

        preprocessed_meta = []
        for i, sequence_k_past in enumerate(sequence_k_pasts):
            sequence_k = sequence_k_past + sequence_q_dim.size
            sequence_k_dim = TensorDim(BlockDimNames.sequence_k, sequence_k)

            tokens = TensorMeta.from_dims(
                hidden_dims[:2], tensor_name=f"tokens_{sequence_k_past}_to_{sequence_k-1}", dtype=torch.int64
            )

            kwargs = {
                **common_kwargs,
                AttentionKwargs.sequence_k_dim: sequence_k_dim,
            }
            if phase != PhaseType.inference:
                kwargs[LanguageModelKwargs.labels] = TensorMeta.from_dims(
                    hidden_dims[:2], tensor_name="labels", dtype=torch.int64
                )
            reference_kwargs = {}
            for name, reference_preprocessed_meta in reference_preprocessed_metas.items():
                reference_tokens, reference_kwargs_ = reference_preprocessed_meta[i]
                for key in (
                    AttentionKwargs.sequence_first,
                    AttentionKwargs.sequence_length,
                    AttentionKwargs.sequence_q_dim,
                    AttentionKwargs.sequence_k_dim,
                ):
                    Assert.eq(reference_kwargs_[key], kwargs[key])
                reference_kwargs[name] = reference_kwargs_
            kwargs["reference_models"] = reference_kwargs

            # TODO ====== Vision ======
            # if self._config.vision_encoder.enabled:
            #     # patch_dimensions are (batch * sequence_length) x 3 x patch_size x patch_size
            #     preprocessed_meta.append((kwargs[VisionEncoderKwargs.image_patches_meta], kwargs))
            # else:
            #     preprocessed_meta.append((tokens, kwargs))

            preprocessed_meta.append((tokens, kwargs))

        return preprocessed_meta

    def preprocess(
        self,
        batch: GPTBatch,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        # TODO ====== Move batch splitting elsewhere, align interface with LayerBase ======
        assert self._is_setup

        if preprocessed_meta is None:
            preprocessed_meta = self.preprocess_meta(batch.token_ids, phase)

        _, common_kwargs = preprocessed_meta[0]
        sequence_q = common_kwargs[AttentionKwargs.sequence_q_dim].size
        sequence_first = common_kwargs[AttentionKwargs.sequence_first]
        prediction_heads: int = self._config.output_layer.prediction_heads

        batch.token_ids = batch.token_ids.to(
            device=self._distributed.device,
            dtype=torch.int64,
            non_blocking=True,
        )

        reference_logits = [{} for _ in preprocessed_meta]
        for name, reference_model in self._reference_models.items():
            reference_preprocessed_meta = [
                (tokens_meta, kwargs_meta["reference_models"][name]) for tokens_meta, kwargs_meta in preprocessed_meta
            ]

            reference_batch = reference_model.fast_llm_model.base_model.preprocess(
                batch, reference_preprocessed_meta, phase=PhaseType.inference, iteration=iteration
            )

            # TODO: Do things work with >1?
            Assert.eq(len(reference_batch), len(preprocessed_meta), 1)
            for i, (reference_tokens, reference_kwargs) in enumerate(reference_batch):
                reference_model.forward(reference_tokens, reference_kwargs, iteration=iteration)
                reference_logits[i][f"{name}_logits"] = reference_kwargs["logits"]

        token_ids = batch.token_ids
        if sequence_first:
            # Move the sequence dimension first to make sequence parallel ops more efficient.
            token_ids = token_ids.transpose(0, 1).contiguous()

        preprocessed = []
        presents = None
        for i, (_, kwargs_meta) in enumerate(preprocessed_meta):
            sequence_k = kwargs_meta[AttentionKwargs.sequence_k_dim].size
            if sequence_first:
                tokens = token_ids[sequence_k - sequence_q : sequence_k]
            else:
                # TODO: Avoid multiple contiguous calls?
                tokens = token_ids[:, sequence_k - sequence_q : sequence_k].contiguous()
            if batch.sequence_lengths is not None:
                kwargs_meta[AttentionKwargs.sequence_lengths] = batch.sequence_lengths
            if batch.chosen_spans is not None:
                kwargs_meta[LanguageModelKwargs.chosen_spans] = batch.chosen_spans
            if batch.rejected_spans is not None:
                kwargs_meta[LanguageModelKwargs.rejected_spans] = batch.rejected_spans

            # TODO: Add pasts/presents to meta input?
            # Use lists as pointers so `past_key_values` is populated during the previous micro_sequence.
            pasts = presents
            presents = None if i == len(preprocessed_meta) - 1 else []
            kwargs = {
                **kwargs_meta,
                AttentionKwargs.past_key_values: pasts,
                AttentionKwargs.presents: presents,
            }
            if phase != PhaseType.inference:
                sequence_offset = sequence_k - sequence_q + 1  # +1 for shift in labels
                if sequence_first:
                    labels = token_ids[sequence_offset : sequence_k + prediction_heads]
                else:
                    # TODO: Avoid multiple contiguous calls?
                    labels = token_ids[:, sequence_offset : sequence_k + prediction_heads].contiguous()
                    # We set label indices to -100 for masked spans, inline with ignore_index in torch.nn.CrossEntropyLoss
                    # TODO: take ignore_index from config
                if batch.loss_masking_spans is not None:
                    # avoid changing input tokens
                    labels = labels.clone()
                    for idx, spans in enumerate(batch.loss_masking_spans):
                        if not spans.numel():
                            continue
                        valid_spans = spans[
                            (spans[:, 0] <= sequence_k + prediction_heads - 1) & (spans[:, 1] >= sequence_offset)
                        ]
                        if valid_spans.numel():
                            # if span is partially within the sequence, truncate parts of spans that are outside of the sequence
                            valid_spans[:, 0].clamp_(min=sequence_offset)
                            valid_spans[:, 1].clamp_(max=sequence_k + prediction_heads - 1)
                            valid_spans -= sequence_offset
                            for start, end in valid_spans:
                                if sequence_first:
                                    labels[start : end + 1, idx] = -100
                                else:
                                    labels[idx, start : end + 1] = -100

                # TODO ====== Preference spans ======
                if batch.chosen_spans is not None:
                    chosen_valid_spans = []
                    for spans in batch.chosen_spans:
                        if not spans.numel():
                            continue
                        # only keep spans within the sequence or partially within the sequence
                        valid_spans = spans[(spans[0] <= sequence_k) & (spans[1] >= sequence_offset)][0]
                        if valid_spans.numel():
                            # if span is partially within the sequence, truncate parts of spans that are outside of the sequence
                            valid_spans[0].clamp_(min=sequence_offset)
                            valid_spans[1].clamp_(max=sequence_k)
                            valid_spans -= sequence_offset

                            chosen_valid_spans.append(valid_spans)
                    kwargs[LanguageModelKwargs.chosen_spans] = chosen_valid_spans

                    rejected_valid_spans = []
                    for spans in batch.rejected_spans:
                        if not spans.numel():
                            continue
                        # only keep spans within the sequence or partially within the sequence
                        valid_spans = spans[(spans[0] <= sequence_k) & (spans[1] >= sequence_offset)][0]
                        if valid_spans.numel():
                            # if span is partially within the sequence, truncate parts of spans that are outside of the sequence
                            valid_spans[0].clamp_(min=sequence_offset)
                            valid_spans[1].clamp_(max=sequence_k)
                            valid_spans -= sequence_offset

                            rejected_valid_spans.append(valid_spans)
                    kwargs[LanguageModelKwargs.rejected_spans] = rejected_valid_spans

                # TODO ====== Vision ======
                # if self._config.vision_encoder.enabled:
                #    if self._config.vision_encoder.image_break_token is not None:
                #        if not labels_cloned:
                #            labels = labels.clone()
                #            labels_cloned = True
                #        labels = torch.where(labels == self._config.vision_encoder.image_break_token, -100, labels)
                #    if self._config.vision_encoder.image_end_token is not None:
                #        if not labels_cloned:
                #            labels = labels.clone()
                #            labels_cloned = True
                #        labels = torch.where(labels == self._config.vision_encoder.image_end_token, -100, labels)
                # Loss-masking for distillation losses
                if self._config.distillation_model is not None:
                    loss_mask = torch.ones_like(labels, dtype=torch.bool)
                    loss_mask = torch.where(labels == -100, False, loss_mask)
                    kwargs[LanguageModelKwargs.loss_mask] = loss_mask
                kwargs[LanguageModelKwargs.labels] = labels
            kwargs.update(reference_logits[i])

            # TODO ====== Vision ======
            # if self._config.vision_encoder.enabled:
            #    batch_images = (
            #        batch.images if batch.images is not None else [[]] * kwargs[AttentionKwargs.micro_batch_size]
            #    )
            #    kwargs[VisionEncoderKwargs.images] = [
            #        [
            #            img.to(device=self._tensor_space.distributed.device, dtype=torch.uint8, non_blocking=True)
            #            for img in images
            #        ]
            #        for images in batch_images
            #    ]
            #    kwargs[VisionEncoderKwargs.image_positions] = (
            #        batch.image_positions
            #        if batch.image_positions is not None
            #        else [[]] * kwargs[AttentionKwargs.micro_batch_size]
            #    )
            #    kwargs[LanguageModelKwargs.tokens] = tokens

            # TODO ====== Turn into super() call ======
            self.embedding.preprocess(tokens, kwargs)
            self.decoder.preprocess(tokens, kwargs)
            self.head.preprocess(tokens, kwargs)

            # TODO ====== Vision ======
            # image_patches = kwargs.get(VisionEncoderKwargs.image_patches, None)
            # if image_patches is not None:
            #     preprocessed.append((image_patches, kwargs))
            # else:
            #     preprocessed.append((tokens, kwargs))

            preprocessed.append((tokens, kwargs))

        return preprocessed

    # TODO ====== Vision ======
    # @property
    # def embedding(self) -> LanguageModelEmbedding:
    #    return self.layers[self.embedding_layer_index]

    # @property
    # def transformer_layers(self) -> list[TransformerBlock]:
    #    return self.layers[self.embedding_layer_index + 1 : -1]

    # @property
    # def embedding_layer_index(self) -> int:
    #    if self._config.vision_encoder.enabled:
    #        return self._config.vision_encoder.transformer.num_layers + 2
    #    else:
    #        return 0

    def get_tied_weights(self) -> dict[str, tuple[ParameterMeta, tuple[int, ...]]]:
        # TODO ====== Tied weights ======
        if self._config.tied_embedding_weight:
            raise NotImplementedError()
        return {}
        # if self._config.output_layer.tied_weight:
        #    return {
        #        WORD_EMBEDDINGS_WEIGHT: (
        #            self.embedding.word_embeddings_weight,
        #            # TODO ====== Vision ======
        #            # (self.embedding_layer_index, *self.model_head_indices),
        #            (0, *self.model_head_indices),
        #        )
        #    }
        # elif self._config.output_layer.prediction_heads > 1:
        #    return {
        #        OUTPUT_WEIGHTS: (
        #            self.model_head.output_weights,
        #            tuple(self.model_head_indices),
        #        )
        #    }
        # else:
        #    return {}

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        return (
            self.embeddings_layer.get_loss_definitions(count)
            + self.decoder.get_loss_definitions(count)
            + self.output_layer.get_loss_definitions(count)
        )


class GPTModel[ConfigType: GPTModelConfig](FastLLMModel[ConfigType]):
    # TODO: Can we drop class?
    pass


class GPTInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[GPTModel]] = GPTModel
    batch_config_class: typing.ClassVar[type[GPTBatchConfig]] = GPTBatchConfig
