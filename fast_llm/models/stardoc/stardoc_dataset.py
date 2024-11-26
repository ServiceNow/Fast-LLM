import os
import io
import logging
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

from fast_llm.data.config import DataConfig
from fast_llm.data.tokenizer import Tokenizer
from fast_llm.data.stardoc_data_utils.docowl_processor import DocProcessor
from fast_llm.data.stardoc_data_utils.utils import (
    convert_queries_and_annotations_to_messages,
    image_loading_function,
)
from fast_llm.data.stardoc_data_utils.docowl_stardoc_processor import docowl_text_preprocess_v1
from fast_llm.data.stardoc_data_utils.constants import IGNORE_INDEX
from fast_llm.engine.distributed.config import PhaseType


logger = logging.getLogger(__name__)


class StarDocDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | None = None,
        im_size: int = 224,
        num_samples: int = -1,
        num_im_tokens: int = 256,
        transforms: bool = False,
        multi_imgs: bool = True,
        split: str = "train",
        tokenizer: Tokenizer | None = None,
        config: DataConfig | None = None,
    ):
        self.im_size = im_size
        self.transforms = transforms
        self.num_samples = num_samples
        self.num_im_tokens = num_im_tokens
        self.multi_imgs = multi_imgs
        self.tokenizer = tokenizer
        phase_map = {
            PhaseType.training: "train",
            PhaseType.validation: "val",
            PhaseType.test: "test",
        }
        self.split=phase_map[split]

        # Use DocOwl processor
        self.processor = DocProcessor(image_size=self.im_size, anchors='grid_9', add_global_img=True, add_textual_crop_indicator=True, media_token=self.tokenizer._config.special_tokens.image_placeholder_token)
        
        assert dataset_path is not None

        # TODO: config validation issue
        multimodal_load_local = True
        if multimodal_load_local:
            # Load from a locally cached copy of the dataset
            self.data_dict = load_from_disk(dataset_path)
            self.data = self.data_dict[self.split]
        else:
            # Load the required spit from HF
            # TODO: configurable cache_dir
            self.data = load_dataset(dataset_path, split=self.split, cache_dir="/mnt/core_llm/cache/", num_proc=os.cpu_count()-1)
        
        if self.num_samples != -1:
            self.data = self.data.select(range(self.num_samples))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        images = sample.get('image', [])

        if self.multi_imgs and not isinstance(images, list):
            images = [images]

        images = image_loading_function(self.data[idx]["image"])

        sample_id = sample["sample_id"]
        dataset_name = sample["dataset_name"]
        queries = sample["queries"]
        annotations = sample["annotations"]
        task_name = sample["task_name"]

        if images[0].size[0] < 10 or images[0].size[1] < 10:
            logger.error("Dummy images with small resolution of < 10x10 seen. Handling these is not implemented")
        
        sample_tokenized_buffer = []
        labels = []

        # Add BOS token at the beginning of the sample
        sample_tokenized_buffer.append(self.tokenizer.bos_token_id)

        # tokenized IDs for "USER:" and "ASSISTANT:"
        user_ids = self.tokenizer.tokenize("USER: ")
        assistant_ids = self.tokenizer.tokenize(" ASSISTANT: ")
        sample_tokenized_buffer.extend(user_ids)

        # Add dummy tokens for all image tokens
        if len(images) > 0:
            # Get all the crops and process them
            all_images, _, processed_query = self.processor(images=images, query=self.tokenizer._config.special_tokens.image_placeholder_token)
            crop_splits = processed_query.split(self.tokenizer._config.special_tokens.image_placeholder_token)[:-1]
            assert len(crop_splits) == len(all_images)
            for crop_split_part in crop_splits:
                sample_tokenized_buffer.extend(self.tokenizer.tokenize(crop_split_part.strip()))
                sample_tokenized_buffer.extend([self.tokenizer.image_placeholder_token_id] * self.num_im_tokens)
        
        # Don't learn on any image tokens
        [labels.append(IGNORE_INDEX) for x in range(len(sample_tokenized_buffer))]

        assert(len(queries) == len(annotations))
        for i, (q, a) in enumerate(zip(queries, annotations)):
            if i>0:
                sample_tokenized_buffer.extend(user_ids)
            sample_tokenized_buffer.extend(self.tokenizer.tokenize(q))
            sample_tokenized_buffer.extend(assistant_ids)
            sample_tokenized_buffer.extend(self.tokenizer.tokenize(a))

        # Add EOS token at the end of the sample
        sample_tokenized_buffer.append(self.tokenizer.eos_token_id)
        labels.extend(sample_tokenized_buffer[len(labels):len(sample_tokenized_buffer)])
        assert len(sample_tokenized_buffer) == len(labels)

        # Right pad to max. sequence length
        n_pad_tokens = self.tokenizer.max_sequence_length - len(sample_tokenized_buffer)
        sample_tokenized_buffer = sample_tokenized_buffer + n_pad_tokens*[self.tokenizer.pad_token_id]

        # Add an extra pad token to the labels at the end to support shifting left 
        labels = labels + (n_pad_tokens + 1) *[IGNORE_INDEX]
        
        return {
            'input_ids': torch.tensor(sample_tokenized_buffer),
            'labels': torch.tensor(labels),
            'images': all_images,
        }