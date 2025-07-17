import concurrent.futures
import os
import random
import struct

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info

# from packed_dataset import PackedDatasetBuilder  # Make sure this is accessible as a module
from tqdm import tqdm
from transformers import AutoTokenizer

# if "HF_HOME" not in os.environ:
#     os.environ["HF_HOME"] = "/mnt/transformers_cache/"
# print(f"HF HOME: {os.environ.get('HF_HOME', 'Not set')}")


# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self, filenames, n_chunks, block_size, seed=12345, shuffle=True, wrap=False, num_processes=1, process_rank=0
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        print(f"[RANK {self._process_rank}] entering PackedDataset iterator", flush=True)
        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
            process_rank=self._process_rank,
        )


class PackedDatasetBuilder:
    def __init__(
        self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None, max_workers=4, parallel_write=False
    ):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []
        self._parallel_write = parallel_write

    def _write_chunk(self):
        if self._parallel_write:
            self._write_chunk_parallel()
        else:
            # Default: synchronous write (original version)
            filename = f"{self._prefix}_{self._counter:010d}.bin"
            filename = os.path.join(self._outdir, filename)
            with open(filename, "wb") as f:
                f.write(HDR_MAGIC)
                f.write(struct.pack("<Q", self._version))
                f.write(struct.pack("<B", code(self._dtype)))
                f.write(struct.pack("<Q", self._chunk_size))
                f.write(self._arr.tobytes(order="C"))
            self._filenames.append(filename)
            self._counter += 1
            self._arr.fill(self._sep_token)
            self._idx = 0

    def _write_chunk_parallel(self):
        # Parallelized version using ThreadPoolExecutor
        filename = f"{self._prefix}_{self._counter:010d}.bin"
        filename = os.path.join(self._outdir, filename)
        arr_copy = self._arr.copy()
        version = self._version
        dtype = self._dtype
        chunk_size = self._chunk_size

        # sep_token = self._sep_token
        def write_task():
            with open(filename, "wb") as f:
                f.write(HDR_MAGIC)
                f.write(struct.pack("<Q", version))
                f.write(struct.pack("<B", code(dtype)))
                f.write(struct.pack("<Q", chunk_size))
                f.write(arr_copy.tobytes(order="C"))
            return filename

        future = self._executor.submit(write_task)
        self._futures.append(future)
        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    def close(self):
        # Wait for all futures to finish
        for future in self._futures:
            future.result()
        self._executor.shutdown()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_reminder(self):
        self._write_chunk()


class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap, process_rank=0):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._process_rank = process_rank

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []
        print(
            f"[RANK] {self._process_rank} file_idx: {self._file_idx}, files: {len(self._filenames)},\
            n_chunks: {self._n_chunks} {self._n_chunks > len(self._filenames[self._file_idx :])}"
        )

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            # if not self._wrap:
            raise StopIteration
            # self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)


# === Config ===
tokenizer_name = "/mnt/checkpoints/diffusion_models/SmolLM2-135M-MASK_TOKEN"
chunk_size = 2**30  # 1GB  # 2**20 # 2**18  # 256K tokens per file
output_dir = "/mnt/datasets/tokenized/SmolLM2-135M/packed_fineweb_sample_350B_largefiles_job"
prefix = "fineweb_sample"
dataset_name = "HuggingFaceFW/fineweb"
dataset_config = "sample-350BT"
split = "train"
input_col = "text"  # The column in the dataset that contains the text data

# === Setup ===
os.makedirs(output_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sep_token = tokenizer.eos_token_id
vocab_size = tokenizer.vocab_size

builder = PackedDatasetBuilder(
    outdir=output_dir,
    prefix=prefix,
    chunk_size=chunk_size,
    sep_token=sep_token,
    vocab_size=vocab_size,
    dtype="auto",
    parallel_write=True,
    max_workers=64,
)

print(f"sep_token: {sep_token}, vocab_size: {vocab_size}")

# === Load and tokenize dataset ===
# stream = load_dataset("HuggingFaceFW/fineweb", name="sample-350BT", split="train", streaming=True,
#                       cache_dir="/mnt/transformers_cache/datasets")
# dataset = []
# num_samples = 0
# max_samles = 100000
# for sample in stream:
#     dataset.append(sample)
#     num_samples += 1
#     if num_samples >= max_samles:
#         break

dataset = load_dataset(
    dataset_name,
    dataset_config,
    split=split,
    trust_remote_code=True,
    num_proc=8,
    cache_dir="/mnt/transformers_cache/datasets",
)

print(f"Dataset loaded: {len(dataset)} samples")

# === Tokenize and pack dataset ===
total_tokens = 0
for ex in tqdm(dataset, desc="Packing dataset"):
    text = ex[input_col]
    if not text.strip():
        continue
    ids = tokenizer(text, add_special_tokens=False).input_ids
    ids.append(sep_token)  # always separate sequences
    builder.add_array(np.array(ids, dtype=builder.dtype))
    total_tokens += len(ids)


# === Finalize ===
builder.write_reminder()
print(f"Total tokens packed: {total_tokens}")
print("Done writing packed dataset.")
print(f"Chunks written: {len(builder.filenames)}")
