"""Epoch dataset schema, importable without training dependencies."""

from fast_llm.config import Field, FieldHint, config_class
from fast_llm.data.dataset.config import SampledDatasetConfig


@config_class(dynamic_type={SampledDatasetConfig: "epoch"})
class EpochDatasetConfig(SampledDatasetConfig):
    _abstract = False
    datasets: list[SampledDatasetConfig] = Field(
        default_factory=list, desc="Prepared sources; each document once per epoch.", hint=FieldHint.core
    )
    plan_summary: dict = Field(
        default_factory=dict, desc="Internal resolved packing plan metadata.", hint=FieldHint.expert
    )

    def build_and_sample(self, config, num_samples, seed):
        from fast_llm.data.dataset.epoch import PlannedEpochDataset, resolve_sources
        from fast_llm.data.dataset.indexed import ConcatenatedDataset

        summary = self.plan_summary
        if not summary:
            raise ValueError("Epoch dataset must be planned by epoch training startup")
        if (
            config.sample_size != summary["identity"]["capacity"]
            or config.sampling_maximum_document_length != summary["identity"]["maximum_document_length"]
        ):
            raise ValueError("Runtime packing capacity/eligibility differs from the epoch plan")
        shards = resolve_sources(self.datasets)
        actual = [
            {
                "path": str(shard.path.resolve()),
                "size": shard.path.stat().st_size,
                "mtime_ns": shard.path.stat().st_mtime_ns,
            }
            for shard in shards
        ]
        if actual != summary["identity"]["shards"]:
            raise ValueError("Epoch shards changed since planning")
        dataset = ConcatenatedDataset("epoch_training", [shard.build() for shard in shards])
        planned = PlannedEpochDataset(dataset, summary)
        if len(planned) != num_samples:
            raise ValueError("Requested training samples differ from the epoch plan")
        return planned
