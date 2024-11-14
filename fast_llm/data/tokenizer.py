from transformers import PreTrainedTokenizerFast

from fast_llm.data.config import TokenizerConfig
from fast_llm.engine.config_utils.run import log_main_rank


class Tokenizer:
    """
    A wrapper around Huggingface (transformers) tokenizer.
    """

    def __init__(self, config: TokenizerConfig, max_sequence_length=None):
        log_main_rank(f"> loading tokenizer from {config.path} ...")
        self._config = config
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.path, errors="replace", max_len=max_sequence_length)
        special_tokens = config.special_tokens.get_special_tokens()
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Token->id mapping for additional special-tokens
        self.special_tokens = {tok: self.tokenizer.vocab[tok] for tok in special_tokens}
        self._inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        self._max_sequence_length = max_sequence_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def max_sequence_length(self):
        return self._max_sequence_length
    
    @property
    def bos_token_id(self):
        bos_token = self._config.special_tokens.bos_token
        if bos_token is not None:
            return self.special_tokens[bos_token]
        else:
            raise ValueError("BOS token not set in tokenizer")
    
    @property
    def eos_token_id(self):
        eos_token = self._config.special_tokens.eos_token
        if eos_token is not None:
            return self.special_tokens[eos_token]
        else:
            raise ValueError("EOS token not set in tokenizer")
    
    @property
    def pad_token_id(self):
        pad_token = self._config.special_tokens.pad_token
        if pad_token is not None:
            return self.special_tokens[pad_token]
        else:
            raise ValueError("PAD token not set in tokenizer")

    @property
    def image_placeholder_token_id(self):
        image_placeholder_token = self._config.special_tokens.image_placeholder_token
        if image_placeholder_token is not None:
            return self.special_tokens[image_placeholder_token]
        else:
            raise ValueError("Image placeholder token not set in tokenizer")

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eos_token_id