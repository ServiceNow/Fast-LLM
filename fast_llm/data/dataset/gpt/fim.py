import numpy as np

from fast_llm.data.dataset.abstract import SampledDataset
from fast_llm.data.dataset.gpt.config import FimConfig, GPTSamplingConfig
from fast_llm.data.dataset.gpt.sampled import GPTSample
from fast_llm.engine.distributed.config import MAX_SEED


class GPTFimDataset(SampledDataset):
    """
    An implementation of FIM (fill in the middle) post-processing of GPT datasets.
    Adapted from https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py
    """

    def __init__(
        self,
        config: FimConfig,
        dataset: SampledDataset,
        sampling_config: GPTSamplingConfig,
    ):
        if sampling_config.use_loss_masking_spans:
            raise NotImplementedError("FIM is currently not compatible with loss masking.")
        self._config = config
        self._dataset = dataset
        self._seed = sampling_config.seed
        self._tokenizer = sampling_config.tokenizer
        if self._tokenizer is None:
            raise ValueError("Fim requires a tokenizer")
        self._suffix_tok_id, self._prefix_tok_id, self._middle_tok_id, self._pad_tok_id = (
            self._tokenizer.vocab[tok]
            for tok in [config.suffix_token, config.prefix_token, config.middle_token, config.pad_token]
        )
        self.fim_split_sample = (
            self._tokenizer.vocab[self._config.split_sample] if self._config.split_sample is not None else None
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> np.ndarray:
        fim_token_ids = self._fim(
            self._dataset[idx].token_ids, np.random.RandomState(seed=(self._seed + idx) % MAX_SEED)
        )
        return GPTSample(fim_token_ids)

    @property
    def name(self) -> str:
        return f"{self._dataset.name}_fim"

    def _fim(self, sample: np.ndarray, np_rng: np.random.RandomState) -> np.ndarray:
        # FIM
        # TODO: permute segments in sample_list, before concatenating.
        sample_len = sample.shape[0]
        eod = self._tokenizer.eod
        segment_breaks = np.argwhere(sample == eod)  # split sample by document

        if segment_breaks.shape != (0, 1):  # then there is an EOD token in this example
            curr_start_position = 0
            new_samples = []
            loc: np.ndarray
            for loc in np.nditer(segment_breaks):
                # Only permute non-empty segments.
                if loc - curr_start_position > 0:
                    # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                    permuted = self._fim_split_and_permute_sequence(sample[curr_start_position:loc], np_rng)
                    new_samples += [permuted, [eod]]

                curr_start_position = loc + 1  # jump over the EOD token
            # Permute the segment after the last EOD
            permuted = self._fim_split_and_permute_sequence(sample[curr_start_position:], np_rng)
            new_samples.append(permuted)

            sample = np.concatenate(new_samples)
        else:
            sample = self._fim_split_and_permute_sequence(sample, np_rng)

        # Truncate or pad sequence to max-length
        diff = sample.shape[0] - sample_len
        if diff > 0:  # too long
            sample = sample[:sample_len]
        elif diff < 0:  # too short
            sample = np.concatenate([sample, np.full((-1 * diff), self._pad_tok_id)])

        assert sample.shape[0] == sample_len
        return sample

    def _fim_split_and_permute_sequence(self, sequence: np.ndarray, np_rng: np.random.RandomState) -> np.ndarray:
        """
        fragment_fim_rate: if set, apply fim with this rate to each fragment.
        """
        if self.fim_split_sample is None:
            return self._fim_permute_sequence(sequence, np_rng, self._config.rate)
        # fim_split_sample is set: split the sample on this token and permute each fragment separately.
        # Typically, if each sample is a repository, then we split again on the file level.
        # Each fragment is a file, and we permute the files.
        fragment_breaks = np.argwhere(sequence == self.fim_split_sample)
        if fragment_breaks.shape == (0, 1):
            # no split token in this sample
            return self._fim_permute_sequence(sequence, np_rng, self._config.rate)
        if not np_rng.binomial(1, self._config.rate):
            # don't do FIM preproc
            return sequence
        # Do FIM on each fragment
        curr_start_position = 0
        new_samples = []
        loc: np.ndarray
        for loc in np.nditer(fragment_breaks):
            if loc - curr_start_position > 0:
                permuted = self._fim_permute_sequence(
                    sequence[curr_start_position:loc], np_rng, self._config.fragment_rate
                )
                new_samples += [permuted, [self.fim_split_sample]]
            curr_start_position = loc + 1  # Jump over the split token
        # Permute the segment after the last split token
        permuted = self._fim_permute_sequence(sequence[curr_start_position:], np_rng, self._config.fragment_rate)
        new_samples.append(permuted)
        return np.concatenate(new_samples)

    def _fim_permute_sequence(
        self,
        sequence: np.ndarray,
        np_rng: np.random.RandomState,
        rate: float,
    ) -> np.ndarray:
        """
        Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
        truncate_or_pad: if True, maintain the same sample length (if transform creates a few extra tokens, drop them).
        """

        if not np_rng.binomial(1, rate):  # sample bernoulli dist
            # don't do FIM preproc
            return sequence

        contents = self._tokenizer.detokenize(sequence)

        # Do not apply FIM if the sample starts with no_fim_prefix
        if self._config.ignore_prefix is not None and contents.startswith(self._config.ignore_prefix):
            return sequence

        if self._config.max_middle_len is None:
            # Sample the two boundaries uniformly at random
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        else:
            # Sample a window-length
            middle_length = np_rng.randint(low=0, high=min(self._config.max_middle_len, len(contents)) + 1)
            first_boundary = np_rng.randint(low=0, high=len(contents) - middle_length + 1)
            # middle_length <= Second-boundary <= len(contents)
            boundaries = [first_boundary, first_boundary + middle_length]

        prefix = contents[: boundaries[0]]
        middle = contents[boundaries[0] : boundaries[1]]
        suffix = contents[boundaries[1] :]

        prefix = np.array([*self._tokenizer.tokenize(prefix, end=False)], dtype=np.int64)
        middle = np.array([*self._tokenizer.tokenize(middle, begin=False, end=False)], dtype=np.int64)
        suffix = np.array([*self._tokenizer.tokenize(suffix, begin=False)], dtype=np.int64)

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if self._config.truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - sequence.shape[0]
            if diff > 0:  # too long
                if suffix.shape[0] <= diff:
                    return sequence
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), self._pad_tok_id)])

        if np_rng.binomial(1, self._config.spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [[self._prefix_tok_id, self._suffix_tok_id], suffix, [self._middle_tok_id], prefix, middle]  # noqa
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [[self._prefix_tok_id], prefix, [self._suffix_tok_id], suffix, [self._middle_tok_id], middle]  # noqa
            )

        return new_sample
