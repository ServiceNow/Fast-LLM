#!/usr/bin/env python3
import pathlib
from fast_llm.data.dataset.gpt.config import (
    GPTSampledDatasetConfig,
    GPTSamplingConfig,
    GPTSamplingData,
)
from fast_llm.engine.distributed.config import PhaseType

def test_dataset_loading():
    # Create minimal config for testing
    dataset_config = {
        "type": "blended",
        "datasets": [
            {
                "type": "concatenated_memmap",
                "path": "/mnt/datasets/tokenized/Mistral-Nemo-Base-2407/dolmino-mix-1124/math/tinyGSM-MIND"
            },
            {
                "type": "concatenated_memmap",
                "path": "/mnt/datasets/tokenized/Mistral-Nemo-Base-2407/starcoderdata/python"
            }
        ],
        "weights": [0.5, 0.5]
    }

    # Create minimal sampling data
    sampling = GPTSamplingData(
        config=GPTSamplingConfig(seed=42),
        sequence_length=2048,
        vocab_size=32000,
        tokenizer=None,
        phase=PhaseType.training,
        distributed=None,
        num_samples=1000,  # Number of samples to process
        cache_directory=pathlib.Path("/tmp/fast_llm_test_cache")  # Temporary cache directory
    )

    try:
        # Build and sample the dataset
        dataset_config = GPTSampledDatasetConfig.from_dict(dataset_config)
        dataset = dataset_config.build_and_sample(sampling)
        
        # Print basic info
        print(f"Dataset type: {type(dataset)}")
        if hasattr(dataset_config, 'datasets'):
            print(f"Number of datasets: {len(dataset_config.datasets)}")
        
        # Try to get a sample
        sample = next(iter(dataset))
        print(f"Sample shape: {sample.shape if hasattr(sample, 'shape') else 'N/A'}")
        
        print("Dataset loading successful!")
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise  # Raising to see the full traceback
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    exit(0 if success else 1)