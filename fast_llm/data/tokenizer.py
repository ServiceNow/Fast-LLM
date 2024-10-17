from transformers import PreTrainedTokenizerFast

from fast_llm.data.config import EOD, TokenizerConfig
from fast_llm.engine.config_utils.run import log_main_rank


class Tokenizer:
    """
    A Huggingface (transformers) tokenizer.
    """

    def __init__(self, config: TokenizerConfig):
        log_main_rank(f"> loading tokenizer from {config.path} ...")
        special_tokens = [EOD]
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.path, errors="replace", max_len=None)
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

class HuggingfacePreTrainedTokenizer:
    """
    A Huggingface (transformers) tokenizer which uses from_pretrained() to load tokenizer
    """

    def __init__(self, config: TokenizerConfig, max_sequence_length: int):
        log_main_rank(f"> loading tokenizer from {config.tokenizer_file} ...")

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)
        # self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
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
    def max_seq_length(self):
        return self._max_sequence_length
    
    @property
    def bos_token_id(self):
        if self.tokenizer.bos_token_id:
            return self.tokenizer.bos_token_id
        else:
            raise ValueError("BOS token not set in tokenizer")
    
    @property
    def eos_token_id(self):
        if self.tokenizer.eos_token_id:
            return self.tokenizer.eos_token_id
        else:
            raise ValueError("EOS token not set in tokenizer")
    
    @property
    def pad_token_id(self):
        if self.tokenizer.pad_token_id:
            log_main_rank("PAD token being set to EOS token")
            return self.tokenizer.pad_token_id
        else:
            return self.tokenizer.eos_token_id

    def tokenize(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def detokenize(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def detokenize_batch(self, token_ids, **kwargs):
        return self.tokenizer.batch_decode(token_ids, **kwargs)

    @property
    def eod(self):
        return self.eod_id


