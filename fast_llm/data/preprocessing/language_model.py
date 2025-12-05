import functools
import logging
import typing

from fast_llm.config import Field, config_class
from fast_llm.data.preprocessing.abstract import NullPreprocessingConfig, PreprocessingConfig
from fast_llm.data.preprocessing.image_patch import ImagePatchConfig
from fast_llm.data.preprocessing.tokenizer import TokenizerConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


@config_class(dynamic_type={PreprocessingConfig: "language_model"})
class LanguageModelPreprocessingConfig(PreprocessingConfig):
    _abstract = False
    tokenizer: PreprocessingConfig = Field()
    # We can't easily compare tokenizers,
    # and in any case the tokenizer path may no longer be valid when loading a prepared dataset,
    # so we provide the vocab size and use it for compatibility checks.
    image_patches: PreprocessingConfig = Field()
    vocab_size: int = Field()
    use_loss_masking_spans: bool = Field(default=False)
    use_preference_spans: bool = Field(default=False)

    def _validate(self) -> None:
        super()._validate()
        Assert.custom(isinstance, self.image_patches, (ImagePatchConfig, NullPreprocessingConfig))
        Assert.custom(isinstance, self.tokenizer, (TokenizerConfig, NullPreprocessingConfig))

    @functools.cached_property
    def use_image_patches(self) -> bool:
        return isinstance(self.image_patches, ImagePatchConfig)

    def check_compatibility(self, preprocessing: typing.Self) -> None:
        Assert.custom(isinstance, preprocessing, LanguageModelPreprocessingConfig)
        # TODO: Check more tokenizer data, ex. bos/eos tokens? path if points to HF hub?
        Assert.geq(self.vocab_size, preprocessing.vocab_size)
        if preprocessing.use_preference_spans:
            # Preference spans are strictly needed for DPO loss.
            assert self.use_preference_spans
        if preprocessing.use_image_patches and self.use_image_patches:
            self.image_patches.check_compatibility(preprocessing.image_patches)
