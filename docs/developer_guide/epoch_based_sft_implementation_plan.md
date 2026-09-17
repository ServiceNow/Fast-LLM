# Implementation plan: epoch-based SFT without manual calculations

## Objective

Keep model, learning rate, output directory, parallelism, sequence length, and other training choices explicit. Automate the calculations currently performed in `Fast-LLM SFT on Gemma Instruct.pdf`: dataset proportions, gradient accumulation, training duration, warmup duration, decay duration, and checkpoint/export steps.

Data preparation remains a separate, existing command. Training builds or loads packing plans internally and immediately proceeds to training. There is no additional resolve command and no required hidden preset.

Reference inputs:

- Guide: `/mnt/queue1/shruthan/docs/Fast-LLM SFT on Gemma Instruct.pdf`
- Example config: `/mnt/queue1/shruthan/experiments/9_14_26_gemma_sft_v4_filtered_dsbench_tgt_only/model_config.yaml`
- Repository: `/mnt/queue1/shruthan/git2/Fast-LLM`

## User-facing workflow and configuration

Continue preparing tokenized data with `fast-llm prepare gpt_memmap --config data_config.yaml`. Continue launching training through `run_fastllm.sh` and the existing `train gpt` command. Add an opt-in epoch mode to the existing GPT trainer configuration rather than requiring a new launcher or training command.

Proposed configuration additions (syntax to finalize against the configuration framework):

```yaml
type: train_gpt
run:
  experiment_dir: /path/to/experiment
pretrained:
  format: gemma4
  model_weights: true
  path: /mnt/queue1/shruthan/models/gemma-4-26B-A4B-it-causal-lm
  load_config: model
model:
  # Retain the existing explicit attention, recomputation, and head settings.
  distributed:
    tensor_parallel: 2
    sequence_data_parallel: 4
    sequence_tensor_parallel: true
    compute_dtype: bfloat16
  multi_stage:
    zero_stage: 3

data:
  micro_batch_size: 32768
  maximum_document_length: 32768
  truncate_documents: false
  datasets:
    training:
      type: epoch
      datasets:
        - type: file
          path: /path/to/sft/fastllm/fast_llm_config.yaml
        - type: file
          path: /path/to/replay/fastllm/fast_llm_config.yaml

training:
  epochs: 3
  global_batch_size: 64
  num_workers: 0
  checkpoint:
    every_epochs: 1
    keep: 1
  export:
    format: gemma4
    every_epochs: 1
  # Existing logging, W&B, and evaluation fields remain available.

optimizer:
  learning_rate:
    base: 2.0e-05
    decay_style: cosine
    warmup_epochs: 0.1
  weight_decay: 0.01
```

The example shows the proposed additions, not a complete replacement for the existing Gemma config. Preserve all explicitly configured model and infrastructure settings.

`data.micro_batch_size` is sequence length in tokens; `training.global_batch_size` counts packed sequences per optimizer step. They have different units. Keep the existing field name for compatibility and explain these units in field descriptions and startup output. Evaluator datasets remain in `data.datasets` unchanged.

In epoch mode, users omit `train_iters`, schedule micro-batch counts, and learning-rate decay iterations. The program derives and displays those values. Reject conflicting explicit and derived settings with field-specific messages; do not silently overwrite user choices. Use `_explicit_fields` to distinguish explicit values from defaults, including an explicitly supplied zero. Do not use zero/sentinel values for conflict detection. In the initial version, `global_batch_size` requires epoch mode and is mutually exclusive with explicit `schedule.depth_first_micro_batches` and `schedule.breadth_first_micro_batches`. Existing iteration-based configs retain their behavior.

## Epoch semantics

The initial implementation supports positive integer epochs, prepared language-model memmap data, and whole-document packing (`truncate_documents: false`). Unsupported combinations receive an explicit error rather than falling back to approximate accounting.

An epoch is one visit to each eligible document across all supplied training sources. Documents exceeding the configured maximum are excluded and reported. Duplicate paths/shards are detected; reject accidental duplicates rather than implicitly treating them as oversampling.

Add `EpochDatasetConfig` to the `SampledDatasetConfig` registry, retaining `data.datasets.training` as the API. It wraps supported source dataset configs and uses the existing sampling/cache/rank plumbing. Resolve standard preparation manifests, which can contain either a single memmap dataset or a token-weighted blend of shards. Flatten these generated shard blends into indexed datasets and concatenate them using `ConcatenatedDataset`. Shuffle documents globally for each epoch. Do not preserve arbitrary weighted sampling while claiming exact dataset passes. Support for custom blends, oversampling, streaming data, truncating packing, or fractional epochs is outside the initial scope.

Global batch size counts packed sequences, not raw documents. An epoch ends with its final packed sequence; documents are never borrowed from the next epoch to fill it.

## Existing integration points

- `fast_llm/data/dataset/indexed.py`: indexed length access and `ConcatenatedDataset`.
- `fast_llm/data/dataset/gpt/config.py`: prepared manifest loading and relative shard-path resolution.
- `fast_llm/data/dataset/sampled.py`: current document shuffling, sparse padded cumulative sums, and sample retrieval.
- `fast_llm/csrc/data.cpp`: `build_padded_token_cumsum`, the current whole-document packing scan.
- `fast_llm/data/data/gpt/config.py` and `data.py`: dataset config, sampling, preprocessing, and iterator construction.
- `fast_llm/data/data/data_loader.py`: distributed sample iteration currently assumes complete batches.
- `fast_llm/engine/training/config.py` and `trainer.py`: duration, initialization, training loop, checkpoints, and exports.
- `fast_llm/engine/schedule/config.py` and `schedule.py`: gradient accumulation and samples per optimizer step.
- `fast_llm/engine/distributed/config.py`: `batch_data_parallel`.
- `fast_llm/engine/optimizer/config.py`: learning-rate configuration; inspect its scheduling implementation before extending it.

New planning code should live under `fast_llm/data/dataset/`, with a small typed result shared by configuration resolution and data loading. Avoid a second independent implementation of packing.

## Phase 1: finite packing planner

1. Add a manifest resolver for the supported generated dataset forms. Reuse existing path-validation rules. Expose underlying indexed datasets without calling `build_and_sample`; the current file wrapper can resolve to a blended config that is not buildable as a single indexed dataset.
2. Read document lengths from the memmap index, not token payloads. Apply eligibility filtering before packing, consistently with runtime retrieval. There is an existing discrepancy: packing filters against sample capacity, while retrieval filters against `min(maximum_document_length, sample_size)`. With the reference settings, a 32,769-token document exposes it. Correct this in the new epoch path and add a regression test; leave legacy iteration-mode behavior unchanged in this work. A legacy correction would be a separate, explicitly documented change.
3. Generate deterministic per-epoch document orders. Keep an explicit seed schedule, choose and record the CPU/GPU permutation backend, and include backend/version information in the plan identity where needed.
4. The existing C++ routine already scans a finite ordered length array. Add a new pybind entry point returning explicit sequence boundaries/counts and padding statistics, sharing internal packing logic where safe. Preserve the existing `build_padded_token_cumsum` signature and iteration-mode behavior. Use `SamplingConfig.sample_size` (`micro_batch_size + predicted_tokens`), rather than assuming the extra prediction token is always one.
5. Store a compact sequence-to-document index and shuffled document IDs. Do not materialize packed token tensors. Define a typed `EpochPackingPlan` with counts, boundaries, seed, eligibility rules, and identity metadata.
6. Flush a partial tail once. Exactly full input must not generate an extra empty sequence; reject an all-ineligible dataset before training. Check integer widths and overflow behavior for large datasets.
7. Implement `PlannedEpochDataset` using the same boundaries for retrieval that were used for counting. Reuse document loading and preprocessing.

Plan outputs include eligible/skipped documents and tokens, sequence count, padding tokens, and packing efficiency. Keep sequence-packing padding distinct from unused slots in the final global batch.

## Phase 2: sequence capacity and resolved configuration

A dataset registry type is the appropriate integration seam, but does not by itself solve startup ordering. `Trainer.__init__` checks `train_iters > 0`, whose current default is zero, and constructs the model. Actual `get_preprocessing_config` currently lives on model instances. We need duration before construction and must not temporarily treat an omitted duration as evaluation-only training.

Planning needs only sequence capacity, not the full preprocessing dictionary. For the supported GPT path, derive `predicted_tokens` from the effective loaded model config's `base_model.head.prediction_heads`; then use `sample_size = micro_batch_size + predicted_tokens`. Add a narrow capacity helper, not a cross-layer preprocessing factory. At dataset sampling/setup, assert the actual instance-derived `predicted_tokens` equals the planned value before consuming the plan. Reject models where capacity cannot be derived from configuration. Do not compare the full preprocessing dictionary: attention preprocessing flags can depend on implementation availability resolved during construction, and those flags do not affect packing capacity. Ensure pretrained model configuration has been loaded before reading prediction heads.

Execution ordering:

1. Validate the user-facing epoch config, including conflicts, without building plans or writing caches. Keep duration-dependent execution checks for the resolved configuration.
2. Resolve topology and config-derived sequence capacity. `micro_batch_splits` affects runtime preprocessing, but not packing capacity or `Schedule.samples_per_batch`; preserve it without including it in a full preprocessing extraction.
3. Coordinate planning using one `Distributed` instance. Build/load plans and distribute the small result metadata.
4. Construct a NEW configuration using `to_copy`/`from_dict`, with derived fields. Validated configs are immutable; neither assignment nor `_set_implicit_default` is a post-validation mutation API. Use separate source and execution configuration types: keep the existing GPT trainer type for the epoch source schema and add an internal resolved GPT execution config subclass. Run user-input mutual-exclusion checks only for the source type. The execution subclass retains epochs/global batch size for display, validates their consistency with derived duration/accumulation, and does not classify derived explicit fields as user conflicts. `to_copy` carries explicit-field information, so explicitly select the execution class through `from_dict` with derived updates rather than expecting `to_copy` to change types. Do not accept the execution type as a public CLI bypass for source validation; resume uses its saved settings only after identity checks.
5. Validate the resolved execution config. Only then call its `get_run(distributed)`, so `config.yaml` contains the actual execution settings, and construct the trainer with positive `train_iters`.
6. Preserve distributed configuration identity/resource ownership when creating the copy. Do not initialize distributed resources twice. Retain the original source config separately for reproducibility.

`EpochDatasetConfig.build_and_sample` receives the existing `SamplingConfig`, including preprocessing, predicted tokens, cache directory, rank, and world size. It loads the matching plan and returns `PlannedEpochDataset`; it must verify requested sample count and packing-relevant identity (including actual predicted tokens), not repack or silently resample. Startup planning and this method call a shared planner/cache API. Extend the call plumbing only where the plan descriptor needs it, rather than introducing an unrelated dataset injection API. Actual instance-derived `predicted_tokens` at setup must match the earlier capacity derivation. Other runtime preprocessing flags are neither needed for planning nor subject to a dictionary-equality check.

Calculations:

```text
batch_data_parallel = world_size / (tensor_parallel * pipeline_parallel * sequence_data_parallel)
accumulation = global_batch_size / batch_data_parallel
steps_in_epoch[e] = ceil(packed_sequences[e] / global_batch_size)
train_iters = sum(steps_in_epoch)
epoch_end_steps = cumulative_sum(steps_in_epoch)
```

Require valid topology and exact accumulation divisibility. Set breadth-first micro-batches to one and derive depth-first micro-batches. Existing iteration-mode scheduling remains unaffected.

For TP=2, PP=1, SDP=4: eight GPUs give one batch replica, so GBS=64 requires accumulation=64; 64 GPUs give eight replicas, requiring accumulation=8.

Set learning-rate decay duration to `train_iters`. Define `warmup_epochs` as a nonnegative duration on the planned epoch timeline: sum complete preceding epoch step counts and floor the fractional next epoch's step count. Zero disables warmup; positive durations produce at least one warmup step. Permit values such as 1.5, bounded by requested epochs. Explicit `warmup_iterations` remains an alternative with mutual exclusion. The example's 0.1 follows the PDF's first-epoch/10 guidance, not the supplied run's 140/7000 settings; do not imply those settings are equivalent. Display resolved steps.

## Phase 3: incomplete final global batches

Loss normalization already aggregates contributing labels/documents across micro-batches and batch-data ranks through `LanguageModelTargetInput.share_batch_data`, called in `ScheduleRunner._preprocess_data`. Reuse that mechanism; do not introduce a new normalization scheme.

Implementation work:

1. Make the planned sample stream contain batch-aligned padding slots at each epoch tail. The current `SampledDatasetIterator` drops incomplete distributed batches. Ensure the planned total is divisible by global batch size, and consequently by batch-data parallelism, so its existing full-batch loop includes the entire tail. Change the iterator only if tests demonstrate that this representation needs it.
2. Support zero-document padding inputs through `TokenBatch.from_documents`, `LanguageModelBatch.from_documents`, optional span/patch/token-data batches, and preprocessing. `TokenBatch` currently indexes `tokens[0]` for padding and cannot directly accept an empty list. Produce valid shape/device/dtype information and zero contributing label/document counts. Verify empty/padding segments work with SDPA/Flash and sequence parallelism; do not assume changing one method is sufficient.

Demote gradient dilution concerns to verification of the existing global normalization. Test padding-only micro-batches/ranks and compare gradients with equivalent unpadded contributing data. An entirely all-masked global batch can also arise from real documents, not just padding; define explicit failure behavior for zero contributing labels rather than allowing NaNs. Keep consumed plan slots separate from real packed sequences in metrics/resume.

## Phase 4: epoch saves, visibility, and resume

Introduce a save-schedule abstraction supporting fixed intervals and explicit step boundaries. Do not merely add `every_epochs` to `IntervalConfig`: `enabled`, `get_count`, `is_sub_interval`, and checkpoint `to_delete`/`keep_every` currently assume modular intervals. Define boundary membership, chronological save ordinals, retention, and compatibility checks for epoch schedules. Reject enabled automated shutdown combined with epoch saves in the first version, unless an explicit boundary-aware shutdown implementation is supplied; preserve W&B alert/log interval checks unchanged. Validate these constraints in both source and execution configs. Guard or replace the unconditional `TrainingConfig._validate` call to `shutdown.assert_sub_interval(self.checkpoint)` when checkpoint scheduling uses explicit epoch boundaries; a rejection rule beside the existing modular assertion is insufficient. For epoch schedules, require shutdown disabled and skip that modular check; retain it for fixed intervals. Preserve existing fixed-interval save behavior. At each requested boundary, checkpoint/export after the optimizer update. Avoid duplicate saves when an epoch boundary is also the final training step. Existing final-save behavior should remain intact.

Write automatically into the experiment directory:

- Original user config.
- Fully resolved execution config.
- Planning summary and plan identity.
- Memory-mappable plan arrays, or references to a shared planning cache.

Do not infer measured efficiency from the reference run's 7,000 iterations: its exact document-pass count and actual packing are not established. The idealized three-pass count is approximately 5,840 optimizer steps at GBS=64 before packing and per-epoch batch rounding; actual counts require planning. Report measured efficiency without imposing an unsupported target range.

Log user settings and derived settings: world size, TP/PP/SDP, batch replicas, accumulation, eligible/skipped documents and tokens, per-epoch packed sequence counts, packing efficiency, per-epoch optimizer steps, total iterations, warmup, and save boundaries.

Fingerprint plans using prepared shard identities, eligibility rules, sequence capacity, seed schedule, packing version, and shuffle backend. Prefer preparation-time immutable dataset IDs/index digests; for older manifests, establish an index identity once and cache it. Avoid hashing all token payloads on every launch.

Use atomic cache writes and an explicit completion marker. One designated rank builds missing plans; other ranks synchronize and load them. Propagate planning errors so other ranks do not wait indefinitely. Cache packing independently of global batch size; derive batch-aligned slots separately.

Store plan identity and completed position with checkpoints. Reuse plans on resume. Initially require unchanged global batch size, topology, and epoch inputs; reject incompatible resumes explicitly.

## Startup cost and measurement

The reference manifests contain 227,666 + 200,000 = 427,666 documents and 4,081,691,682 tokens. Three epochs require approximately 1.28 million document placements, not scans of four billion token values.

Approximate array sizes if using int32 document IDs/lengths:

- Document lengths: 1.7 MB.
- Three document permutations: 5.1 MB.
- Sequence boundaries and metadata: additional space proportional to the packed sequence count.

Complexity is O(documents * epochs) for shuffle/packing. Read length metadata once. Build epochs sequentially if limiting peak memory. Memory mapping token files does not imply eagerly reading their entire payload.

Expected cold-start planning cost is seconds to tens of seconds at this dataset size on responsive storage, but this is an unmeasured estimate. Shared-storage latency, shard count, process initialization, and synchronization can dominate. Cached-plan reuse should be faster. Model loading and raw-data preparation are separate costs.

Instrument manifest opening, length reads, filtering, shuffling, packing, cache writes/reads, and synchronization. Benchmark cold and warm launches on the actual datasets; publish measured timings in the change description. If metadata reads dominate, extend preparation to emit a compact document-length sidecar. Sequence-specific plans should stay in training because sequence length and seeds are training choices.

## Validation and acceptance criteria

### Focused tests

- One document, exactly full sequence, partial tail, multiple padded sequences, and all-ineligible input.
- Documents near the maximum-length and prediction-token boundaries.
- Every eligible document appears exactly once per epoch across multiple shards/sources.
- Retrieved sequences match planned membership and packing counts.
- Deterministic plan generation and cache reuse; changes in relevant inputs invalidate the cache.
- Batch calculations with TP/PP/SDP, explicit/default conflict detection, incompatible requested batch sizes, and orthogonal micro-batch splits.
- Config-derived prediction-head count versus instance-derived `predicted_tokens`, including multiple prediction heads where supported; unsupported capacity sources fail clearly.
- Source-only conflict checks and execution consistency checks, preserving explicit-field information across reconstruction.
- Resolved run serialization and positive duration before trainer construction.
- Epochs with different sequence counts, fractional/multi-epoch warmup, correct cumulative save boundaries, retention ordinals, and shutdown incompatibility validation.
- Correct partial-batch loss/gradients, including entirely padding-only micro-batches/ranks.
- Checkpoint/resume produces the same subsequent document sequence and updates as uninterrupted execution.
- Existing iteration-based configs and fixed-interval saves retain behavior.

### Integration validation

Run a small multi-shard masked-SFT dataset through a short distributed training job, epoch checkpoints, resume, and Gemma-format export. Then benchmark planning on the reference manifests and run a short real-model smoke test using the explicit Gemma settings.

Acceptance criteria:

1. User runs only preparation and the existing training launch.
2. Model, learning rate, output, TP, SDP, sequence length, and other expert settings remain visible.
3. User supplies epochs and global batch size without manually calculating token weights, accumulation, train iterations, warmup, decay, or save steps.
4. Step counts come from the same packing plans used by training; no 5% adjustment is used.
5. Each eligible document is consumed once per epoch without repetition to fill final batches.
6. Startup timings and all derived values are visible and saved.
7. Resume is reproducible and incompatible changes are detected.

## Delivery order

1. Finite packing planner and planned-dataset retrieval, with boundary and coverage tests.
2. Final-global-batch padding and verified loss normalization.
3. Epoch dataset registry type, config-level sequence-capacity helper, resolved config copying, and pre-trainer execution hook.
4. Epoch save boundaries, cache coordination, visibility, and resume metadata.
5. Distributed integration validation and real-data startup benchmark.
6. Example config migrated from the supplied Gemma config and user documentation.

Ship the complete exact-epoch path together. Do not expose a configuration that promises complete epochs while still using approximate weighted sampling or dropping tail batches.

## Review disposition

The implementation plan was revised after checking the supplied review against the repository. Accepted corrections include the dataset registry API, validated-config immutability, explicit-field conflict detection, preprocessing startup dependency, existing batch-global loss normalization, legacy eligibility mismatch, and interval/retention integration.

Two review suggestions are qualified: a dataset type alone cannot provide duration before trainer construction without addressing sequence capacity; and 7,000 configured steps do not prove a specific packing overhead. Empty-batch support also spans the base token batch and preprocessing, so its cost should be established through tests rather than assumed trivial.

## Implementation status

Implemented in the `shruthan/epoch-sft` worktree: finite C++ packing boundaries, cached CPU epoch permutations, dataset-registry integration, narrow prediction-token capacity verification, resolved execution config reconstruction, partial-batch padding using existing global loss counts, epoch saves/retention, checkpoint identity checks, and source/resolved config artifacts. Existing iteration packing is unchanged.

The implementation uses `fast_llm/data/dataset/epoch_config.py` for the lightweight schema and `epoch.py` for runtime planning/loading. Runtime input targets enable zero-supervision rejection only in epoch mode. Save configs implement boundary membership and chronological ordinals directly; modular shutdown checks are guarded for epoch schedules.

Plans use int64 arrays in this version (larger than the illustrative int32 estimate above). Prepared dataset identity uses canonical paths, size/mtime, and a document-length digest, assuming prepared datasets are immutable; it does not hash all token payloads. Micro-batch splits are preserved in preprocessing and checked in execution identity, but do not affect packing capacity.

Measured three-epoch planning on the reference sources: 3.59 seconds cold, 0.03 seconds cached, approximately 80.3% packing efficiency, and 6,470 total steps at GBS=64. See `docs/recipes/epoch-based-sft.md` and `examples/gemma4_epoch_sft.yaml` for usage. Tiny two-GPU SDP/SDPA and DP/Flash runs cover padding-only slots/ranks, epoch exports/checkpoints, and exact-weight resume reproduction. Full Gemma training has not been launched.

Fractional checkpoint/export cadence is supported via positive finite `every_epochs` values. Save points are rounded forward to completed optimizer steps on each planned epoch timeline, deduplicated, and include the final step. Checkpoint and export cadence/retention remain independent.
