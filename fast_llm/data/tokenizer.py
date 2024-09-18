from transformers import PreTrainedTokenizerFast

from fast_llm.data.config import EOD, TokenizerConfig
from fast_llm.run import log_main_rank


class Tokenizer:
    """
    A Huggingface (transformers) tokenizer.
    """

    def __init__(self, config: TokenizerConfig):
        log_main_rank(f"> loading tokenizer from {config.tokenizer_file} ...")
        special_tokens = [EOD]
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.tokenizer_file, errors="replace", max_len=None)
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.eod_id = self.tokenizer.vocab[EOD]
        # Token->id mapping for additional special-tokens
        self.special_tokens = {tok: self.tokenizer.vocab[tok] for tok in special_tokens}
        self._inv_vocab = {v: k for k, v in self.tokenizer.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
