import numpy as np
from fast_llm.diffullama.packed_dataset import PackedDatasetBuilder
from tqdm import tqdm

def pack_worker(block, builder_args, start_counter):
    builder = PackedDatasetBuilder(**builder_args)
    builder._counter = start_counter  # Set starting file number for this worker
    total_tokens = 0
    for ex in tqdm(block, desc="Packing block of tokenized dataset"):
        if not ex["ids"]:
            continue
        builder.add_array(np.array(ex["ids"], dtype=builder.dtype))
        total_tokens += len(ex["ids"])
    builder.write_reminder()
    return builder.filenames, total_tokens
