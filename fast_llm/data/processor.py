from fast_llm.data.tokenizer import Tokenizer


class MultiModalProcessor:
    """
    Combines multiple modalities (text and image) and converts to tokens/patches for text and images.
    """

    def __init__(self, tokenizer: Tokenizer, image_processor=None):
        self._tokenizer = tokenizer
        self._image_processor = image_processor
