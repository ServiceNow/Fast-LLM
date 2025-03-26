import datasets

from fast_llm.data.preparator.hf_processors.configs.doc_length_filter import DocLengthFilterConfig

def apply(config: DocLengthFilterConfig, dataset: datasets.Dataset) -> datasets.Dataset:
    # do dataset.filter eliminating too long or too short docs
    return dataset