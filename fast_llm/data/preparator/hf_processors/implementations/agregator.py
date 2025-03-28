import datasets

from fast_llm.data.preparator.hf_processors.configs.agregator import AgregatorConfig

def apply(config: AgregatorConfig, dataset: datasets.Dataset) -> datasets.Dataset:
    # do something before applyting each processor
    for step in config.steps:
        dataset = step.apply(dataset)
        # compute metrics
    # save meterics, from all ranks?
    return dataset