import datasets
import pathlib
import time
import typing

import torch
import torch.distributed

from fast_llm.data.preparator.gpt_memmap.distributed_config import DatasetPreparatorDistributedConfig


class ProcessorMetricsLogger:
    def __init__(
        self, distributed_config: DatasetPreparatorDistributedConfig, field: str, num_proc: int, batch_size: int
    ):
        self.start_time = None
        self.distributed_config = distributed_config
        self.field = field
        self.num_proc = num_proc
        self.batch_size = batch_size
        self.local_times = []
        self.local_doc_lengths = []
        self.local_chars = []

    def start(self):
        self.start_time = time.time()

    def stop(self, dataset: datasets.Dataset, step_name: str):
        # TODO: seems generated nonsense, rewrite manually
        elapsed_time = time.time() - self.start_time
        num_rows = len(dataset)

        def compute_doc_lengths(batch):
            return {"doc_lengths": [len(doc) for doc in batch[self.field]]}

        doc_lengths = dataset.map(
            compute_doc_lengths, batched=True, batch_size=self.batch_size, num_proc=self.num_proc
        )
        doc_lengths = sum(doc_lengths["doc_lengths"], [])
        num_chars = sum(doc_lengths)

        self.local_times.append(elapsed_time)
        self.local_doc_lengths.extend(doc_lengths)
        self.local_chars.append(num_chars)

        local_stats = torch.tensor(
            [num_rows, num_chars, min(doc_lengths, default=0), max(doc_lengths, default=0)], dtype=torch.long
        )
        all_stats = [
            torch.zeros_like(local_stats) for _ in range(torch.distributed.get_world_size(self.process_group))
        ]

        if torch.distributed.is_initialized():
            torch.distributed.all_gather(all_stats, local_stats, group=self.process_group)

        if self.rank == 0:
            all_times = torch.tensor(self.local_times)
            all_chars = torch.tensor(self.local_chars)
            min_time, max_time, avg_time = all_times.min().item(), all_times.max().item(), all_times.mean().item()
            min_chars, max_chars, total_chars = all_chars.min().item(), all_chars.max().item(), all_chars.sum().item()
            min_doc_length = min(stat[2].item() for stat in all_stats)
            max_doc_length = max(stat[3].item() for stat in all_stats)
            total_rows = sum(stat[0].item() for stat in all_stats)

            return {
                "step_name": step_name,
                "elapsed_time": {"min": min_time, "max": max_time, "avg": avg_time},
                "document_length": {"min": min_doc_length, "max": max_doc_length, "total": total_chars},
                "total_rows": total_rows,
            }
        return None

    @classmethod
    def format(cls, metrics: dict[str, typing.Any]):
        return (
            f"Processor {metrics['step_name']}' applied, max shard processing time {metrics['elapsed_time']['max']},"
            f" number of rows remained in the dataset {metrics['total_rows']},"
            f" number of characters remained in the dataset {metrics['document_length']['total']}"
        )

    @classmethod
    def save_as_yaml(cls, file_name: pathlib.Path, metrics: list[dict[str, typing.Any]]):
        import yaml

        with file_name.with_suffix(".yaml").open("wt") as f:
            yaml.safe_dump(metrics, f)
