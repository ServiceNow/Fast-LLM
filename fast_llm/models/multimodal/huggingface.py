import functools
import logging
import re
import typing

import torch
import transformers.modeling_outputs

from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelInput
from fast_llm.data.document.patch import PatchBatch
from fast_llm.data.preparation.image_patch import ImagePreparationConfig
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelConfig, HuggingfaceGPTModelForCausalLM
from fast_llm.models.multimodal.config import MultiModalModelConfig
from fast_llm.models.multimodal.model import MultiModalBaseModel, MultiModalInferenceRunner, MultiModalModel
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class HuggingfaceMultiModalModelConfig(HuggingfaceGPTModelConfig):
    model_type = "fast_llm_multi_modal"
    model_config_class = MultiModalModelConfig

    # transformers v5: PretrainedConfig is a dataclass, so redefining a field in a subclass
    # would create a new dataclass field with a different default. Guard with TYPE_CHECKING
    # so type checkers see the narrowed type without affecting the runtime dataclass layout.
    if typing.TYPE_CHECKING:
        fast_llm_config: MultiModalModelConfig


class HuggingfaceMultiModalModelForCausalLM(HuggingfaceGPTModelForCausalLM):
    config_class = HuggingfaceMultiModalModelConfig
    config: HuggingfaceMultiModalModelConfig
    runner_class: typing.ClassVar[type[MultiModalInferenceRunner]] = MultiModalInferenceRunner
    fast_llm_base_model: MultiModalBaseModel

    def __init__(
        self,
        fast_llm_model: MultiModalModel,
        config: HuggingfaceMultiModalModelConfig | None = None,
        runner: ScheduleRunner | None = None,
        **kwargs,
    ):
        super().__init__(fast_llm_model, config, runner, **kwargs)
        embedding_config = self.config.fast_llm_config.base_model.vision_encoder.embeddings
        self._patch_config = ImagePreparationConfig(
            height=embedding_config.patch_height,
            width=embedding_config.patch_width,
            do_resize=False,
        )
        self._image_token_index = self.config.fast_llm_config.base_model.image_token_index
        assert self._image_token_index is not None

    def inner_forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | transformers.modeling_outputs.CausalLMOutputWithPast:
        return self._inner_forward(
            self._get_batch(input_ids, attention_mask, pixel_values, image_sizes),
            input_ids.shape,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def _get_batch(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_sizes: torch.Tensor | None = None,
    ):
        batch = super()._get_batch(input_ids, attention_mask)
        if pixel_values is None:
            images = []
        elif image_sizes is None:
            images = pixel_values.unbind()
        else:
            # Hugging Face uses a batch of padded images with shape (num_images, max_height, max_width)
            # We need to remove padding before further processing.
            images = [image[:, :height, :width] for image, (height, width) in zip(pixel_values, image_sizes)]

        # Convert to patches. TODO: Creating token map and image token ids unnecessarily.
        image_patches, image_position_ids, _, _, patch_counts = self._patch_config.get_patches_from_images(images)

        # Hugging Face encodes token positions through an image token, from which we extract the patch mapping.
        image_mask = batch.tokens == self._image_token_index

        (token_map,) = torch.nonzero(image_mask, as_tuple=True)

        Assert.eq(len(token_map), len(image_patches))
        # Fast-LLM uses negative token ids as placeholders for image tokens.
        batch.tokens = torch.where(image_mask, -100, batch.tokens)

        batch.image_patches = PatchBatch(
            patches=image_patches,
            token_map=token_map,
            positions=image_position_ids,
            lengths=patch_counts,
        ).to_device_(input_ids.device)

        return batch

    def _get_input(
        self,
        batch: LanguageModelBatch,
        past_key_values=None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> LanguageModelInput:
        model_input = super()._get_input(batch, past_key_values, use_cache, output_hidden_states)
        if output_hidden_states and isinstance(output_hidden_states, bool):
            model_input.output_hidden_states.update(
                re.compile(pattern)
                for pattern in (
                    self.fast_llm_base_model.vision_encoder.embeddings.module_name + "$",
                    *(layer.module_name + "$" for layer in self.fast_llm_base_model.vision_encoder.encoder),
                    self.fast_llm_base_model.vision_encoder.adapter.module_name + "$",
                )
            )
        return model_input

    @functools.cached_property
    def preprocessing_config(self):
        preprocessing_config = self._fast_llm_model.get_preprocessing_config(PhaseType.inference)
        return preprocessing_config.from_dict(preprocessing_config, {("vision_encoder", "normalization"): None})
