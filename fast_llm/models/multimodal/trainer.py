import logging
import typing

from fast_llm.data.preprocessing.language_model import LanguageModelPreprocessingConfig
from fast_llm.models.gpt.trainer import GPTTrainer
from fast_llm.models.multimodal.config import MultiModalTrainerConfig

logger = logging.getLogger(__name__)


class MultiModalTrainer[ConfigType: MultiModalTrainerConfig](GPTTrainer[ConfigType]):
    def _get_preprocessing_config(
        self, *, _return_dict: bool = False
    ) -> LanguageModelPreprocessingConfig | dict[str, typing.Any]:
        out = super()._get_preprocessing_config(_return_dict=True)
        out["image_patches"] = {
            "height": self._config.model.base_model.vision_encoder.embeddings.patch_height,
            "width": self._config.model.base_model.vision_encoder.embeddings.patch_width,
            # TODO: Max shape and special tokens are unspecified in the model.
            "max_image_height": 2**32,
            "max_image_width": 2**32,
            "image_break_token": None,
            "image_end_token": None,
        }
        return out if _return_dict else LanguageModelPreprocessingConfig.from_dict(out)
