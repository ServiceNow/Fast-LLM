import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from fast_llm.data.config import TokenizerConfig
from fast_llm.engine.config_utils.run import log_main_rank


class Tokenizer:
    """
    A wrapper around Huggingface (transformers) tokenizer.
    """

    def __init__(self, config: TokenizerConfig):
        log_main_rank(f"> loading tokenizer from {config.path} ...")
        self.tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=config.path, errors="replace", max_len=None
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not have an EOS token.")
        self.eod_id = self.tokenizer.eos_token_id
        self._inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def vocab(self) -> dict[str, int]:
        return self.tokenizer.vocab

    @property
    def inv_vocab(self) -> dict[int, str]:
        return self._inv_vocab

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        add_bos_token: bool | None = None,
        add_eos_token: bool | None = None,
    ) -> list[int]:
        # add_special_tokens will use the default tokenizer behaviour.
        # If add_bos_token or add_eos_token is set, we use them and ignore add_special_tokens.
        if add_bos_token is not None or add_eos_token is not None:
            return (
                ([self.tokenizer.bos_token_id] if add_bos_token and self.tokenizer.bos_token_id else [])
                + self.tokenizer.encode(text, add_special_tokens=False)
                + ([self.tokenizer.eos_token_id] if add_eos_token else [])
            )
        else:
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def detokenize(self, token_ids: int | list[int] | np.ndarray | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
