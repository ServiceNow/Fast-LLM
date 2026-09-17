# Epoch-based SFT

Prepare each source with the existing `fast-llm prepare gpt_memmap` command. Then run training normally:

```bash
python -m torch.distributed.run --nproc_per_node=8 -m fast_llm.cli train gpt --config examples/gemma4_epoch_sft.yaml
```

The example retains explicit Gemma model, attention, learning rate, output, and parallelism settings. Update paths and launch topology for your environment. There is no resolve command.

## Inputs and derived settings

Set `data.datasets.training.type: epoch`, with prepared `type: file` manifests under its `datasets` list. Set `training.epochs` and `training.global_batch_size`.

`data.micro_batch_size` is sequence length in tokens. `training.global_batch_size` counts packed sequences per optimizer update. With TP=2 and SDP=4, eight GPUs process one independent sequence at a time: GBS=64 needs 64 accumulation micro-batches. With 64 GPUs, it needs eight.

Do not supply `training.train_iters`, schedule depth/breadth micro-batch counts, or learning-rate decay iterations. They are derived from the packing plan. Existing iteration-based configs continue to work.

Use `optimizer.learning_rate.warmup_epochs` to express warmup in epochs, or explicit `warmup_iterations`. By default warmup lasts 0.1 epoch. Multi-epoch warmup sums the planned lengths of complete epochs plus the requested fraction of the next one.

Checkpoints and exports default to each epoch plus the final step. Their `every_epochs` settings are independent. To save training checkpoints every quarter epoch while exporting only each full epoch:

```yaml
training:
  checkpoint:
    every_epochs: 0.25  # Use 0.5 for half epochs.
    keep: 4
  export:
    every_epochs: 1
    format: gemma4
```

Fractions are mapped to the first completed optimizer step reaching each save point, using each epoch's actual step count. When several points land on one step, save once. Always save at the final training step. `keep` controls retained checkpoints, not save frequency.

Set `every_epochs` to change the frequency. Explicit iteration `interval` settings remain available as an alternative; `interval: null` disables that save. Automated shutdown is unsupported with epoch checkpoint schedules.

## Exact duration

The planner filters ineligible documents, shuffles each epoch deterministically on CPU, and packs whole documents without truncating them. It concatenates source/shard documents; each eligible document appears once per epoch. Weighted source oversampling is not provided in this mode. Long documents are skipped and reported.

For each epoch, optimizer steps equal `ceil(packed_sequence_count / global_batch_size)`. The final batch uses zero-loss padding slots rather than dropping or repeating examples. A global batch with no supervised labels fails explicitly.

The initial version requires integer epochs, prepared single-file memmap data, `truncate_documents: false`, epoch shuffling, and supervised label losses without reference models. Raw data remains handled by preparation.

## Startup and resume

Planning runs automatically before trainer construction. Cache arrays contain shuffled document IDs and sequence boundaries, not copied token payloads. Rank zero builds missing plans; other ranks load them after receiving the resolved metadata. Runtime prediction-token capacity is checked against the plan.

The output directory contains `sft_config.yaml` (source configuration), `config.yaml` (resolved execution settings), `epoch_plan.json` (counts, efficiencies, identities, and timing), and cached plans under `dataset_cache/epochs/`.

Resume with the same source config and output directory. Checkpoints store epoch execution identity. Changed dataset identity, epochs, seeds, sequence rules, batch size, parallel topology, or micro-batch splits are rejected; use a new directory for those changes. Do not use the internal resolved execution config as a new source config.

## Measured planning cost

On the supplied two-source dataset (427,666 documents), planning three epochs took 3.59 seconds on the current machine; cached reuse took 0.03 seconds. These measurements exclude interpreter/import startup, distributed initialization, tokenization, and model loading. Storage and hardware affect timings.

At sequence length 32,768 and GBS=64, the planner found 416,945 eligible documents and excluded 10,721 over-length documents (450,848,696 tokens). It produced 138,047 / 137,983 / 138,016 sequences and 2,157 / 2,156 / 2,157 optimizer steps: 6,470 steps total. Measured packing efficiency was approximately 80.3%.

Rebuild the C++ extension after updating the source (`python setup.py build_ext --inplace`, or the repository's usual installation/image build). The new finite packing entry point is independent of the legacy iteration sampler.
