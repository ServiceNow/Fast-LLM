# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation

```bash
# Full install with GPU support (requires CUDA)
pip install -e ".[CORE,OPTIONAL,DEV]"

# CPU-only install (for IDE support, no GPU required)
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE FLASH_ATTENTION_FORCE_BUILD=TRUE pip install -e ".[CORE,DEV]" --no-build-isolation
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

Hooks run Black (line length 119), isort, autoflake, and pyupgrade automatically on commit.

### Running tests

Always redirect output to a file to avoid truncation, e.g. `pytest ... 2>&1 | tee /tmp/fast_llm_tests/pytest_out.txt`. Use `/tmp/fast_llm_tests/` as the output directory (create it if needed).

```bash
# All tests
pytest -v -n 6 tests/

# Single test file or function
pytest -v tests/layers/test_lm_losses.py
pytest -v tests/layers/test_lm_losses.py::test_name

# Run extra-slow tests (disabled by default)
pytest -v -n 6 --run-extra-slow tests/

# Filter by model type
pytest -v -n 6 --models gpt tests/

# Test Triton kernels on CPU (no GPU required)
TRITON_INTERPRET=1 pytest -v tests/layers/test_lm_losses.py
```

The test suite sets `FAST_LLM_SKIP_TRITON_AUTOTUNE=TRUE` automatically. Tests that require distributed execution spawn child processes via `torchrun`. `TRITON_INTERPRET=1` enables the Triton interpreter so Triton kernels run on CPU — use this when developing or debugging Triton code without a GPU.

When working with external models, run as a **separate** pytest invocation — combining `tests/` and `fast_llm_external_models/tests/` in one run causes OOM:

```bash
pytest -v -n 6 fast_llm_external_models/tests/
```

Tests in `tests/models/` chain via pytest-depends and must be run as a whole file. Use `--models <name>` to filter; never use `-k` or `::test_name`, which breaks the dependency chain (causes "dependency not found" failures).

### CLI

```bash
# General form
fast-llm <subcommand> [--config config.yaml] [key=value overrides...]

# Validate config without running
fast-llm train gpt --config config.yaml --validate

# Example: train GPT
fast-llm train gpt --config examples/mistral-4-node-benchmark.yaml
```

## Design principles

- **Generalize rather than special-case.** New features should extend existing abstractions, not create parallel ones for a specific use case. If `Attention` doesn't cover a new model variant, extend its config rather than introducing `MyModelAttention`. Same principle for losses, MLP variants, normalization layers — prefer parameterizing the existing module over forking it.

## Architecture

### Configuration system (`fast_llm/config.py`)

The core infrastructure. Every config is a frozen dataclass decorated with `@config_class()` that inherits from `Config`. Fields use `Field(default=..., desc=..., hint=FieldHint.X)` with hints that control serialization verbosity and validation:

- `FieldHint.architecture` — defines model structure; compared across checkpoints
- `FieldHint.core` — always required explicitly
- `FieldHint.optional/performance/stability/feature/expert` — optional tuning knobs
- `FieldHint.derived` — computed from other fields, never serialized

Dynamic dispatch (for YAML `type:` keys) uses `@config_class(dynamic_type={BaseClass: "name"})`. The registry enables subclass selection from config files.

`RunnableConfig` (in `fast_llm/engine/config_utils/runnable.py`) extends `Config` with CLI parsing. `fast-llm train gpt` chains two levels of dynamic type dispatch: `train` selects the trainer subcommand, `gpt` selects `GPTModelConfig`.

**Important:** Config modules (`config.py` files) must not import heavy third-party packages (torch, numpy, etc.) at the top level — only barebone dependencies — so configs can be validated without a full GPU environment.

### Engine (`fast_llm/engine/`)

The training engine is model-agnostic. Key components:

- **`distributed/`** — `DistributedConfig` defines tensor/pipeline/data/sequence parallelism. `Distributed` manages NCCL process groups and knows which ranks are peers in each dimension.

- **`base_model/`** — `BaseModel` (abstract) and `LayerBase` define the layer interface. A model is a flat list of `Layer` objects returned by `get_layers()`. Layers are the unit of pipeline parallelism.

- **`multi_stage/`** — `MultiStageModel` wraps a `BaseModel` and handles:
  - Splitting layers across pipeline stages
  - ZeRO-1/2/3 weight/gradient/optimizer-state sharding via `FSDP`
  - Tied parameter management across stages

- **`schedule/`** — `Schedule` builds the micro-batch execution graph; `ScheduleRunner` executes it, orchestrating pipeline-parallel forward/backward passes with gradient accumulation.

- **`optimizer/`** — AdamW implementation in `fast_llm/functional/triton/adam.py`; `Optimizer` manages `ParamGroup`s with per-group LR schedules.

- **`training/`** — `Trainer` base class wires everything together. Subclasses (e.g., `GPTTrainer`) provide model-specific data loading. Training loop, checkpointing, evaluation, and W&B logging are handled here.

- **`checkpoint/`** — Supports Fast-LLM distributed format, safetensors, and HuggingFace format conversion.

### Layers (`fast_llm/layers/`)

Reusable building blocks consumed by models:

- `common/` — `Linear` (with tensor-parallel variants), normalization (LayerNorm, RMSNorm), PEFT (LoRA)
- `attention/` — Multi-head/grouped-query attention, RoPE embeddings
- `decoder/` — `TransformerBlock` composing attention + MLP, various MLP variants (dense, MoE)
- `language_model/` — `LanguageModelEmbedding`, `LanguageModelHead`, loss functions (CE, entropy, z-loss, DPO, GRPO)
- `ssm/` — State space model layers (Mamba)
- `vision/` — Vision encoder layers for multimodal models

### Models (`fast_llm/models/`)

Concrete model implementations:

- `gpt/` — The main model family. `GPTBaseModel` assembles embedding + decoder blocks + LM head. `GPTModelConfig` registers HuggingFace checkpoint converters for Llama, Mistral, Mixtral, Qwen2, etc. `GPTTrainer` is the entry point for `fast-llm train gpt`.
- `multimodal/` — Vision-language model built on top of GPT.

### Functional / Triton kernels (`fast_llm/functional/`)

Low-level ops with optional Triton acceleration. Triton kernels live in `fast_llm/functional/triton/` and fall back to PyTorch when unavailable. Key kernels: fused entropy loss, z-loss, Adam, sparse linear (MoE), GRPO loss.

`fast_llm/functional/triton/__init__.py` is the Triton entry point — it handles import errors, exposes `triton_available`/`triton_interpret` flags, and contains workarounds for Triton interpreter bugs. If a third-party Triton bug needs fixing, monkeypatch it here rather than editing `third_party/`.

**`third_party/` is read-only.** Never edit files under `third_party/`. Fix issues by monkeypatching the relevant module attribute in Fast-LLM code (typically `fast_llm/functional/triton/__init__.py`).

### Data (`fast_llm/data/`)

- `dataset/` — Memmap, blended, concatenated, FIM, random, and streaming datasets
- `data/gpt/` — GPT-specific data pipeline (tokenized memmap sequences)
- `preparation/` — Offline dataset preprocessing tools
- `document/` — Document-level abstractions for variable-length inputs

## Testing Conventions

Tests live in `tests/`. The following patterns work well in this codebase.

**Structure:**
- Prefer thin test bodies: construct inputs, call the function, compare outputs. Put expected-value derivation in a helper dataclass with `@cached_property` fields built up step by step.
- Return `None` from an `expected_*` property when a feature flag is off so the test body stays unconditional.
- Test all outputs, not just `[0]`. When a function returns results indexed by multiple dimensions (e.g. splits × targets), loop over every result; structure expected values to mirror the loop.
- Build complex expected values through layered named `cached_property` fields, each adding one transformation. A test failure then points to the layer where it diverges.

**Parametrization:**
- Generate test cases as a cross-product of `base_cases × variants` via list comprehension with a `_make_name` helper and a filter clause for invalid combinations.
- Include boundary inputs (e.g., sequences shorter than a parameter, zero padding) as named base cases with explanatory comments.
- Prefer adding specific named cases over new parameter dimensions — cross-products get costly fast. Expand a dimension only when there's a real reason.

**Reference implementations:**
- Reference/ground-truth functions in tests must stay independently correct. Never change a reference to match new implementation behavior — if they disagree, suspect the new implementation first.
- Prefer plain Python loops over tensor ops in reference helpers to stay clearly independent from the implementation.

**Result paths:**
- Tests write debug artifacts (logs, configs, intermediate state) to a unique named subdirectory under `result_path`. This material is valuable for post-mortem investigation of failures — preserve the per-test subdirectory pattern.

## Code Style

- **Imports**: Third-party → `import package.module` (keep fully qualified). First-party → `from fast_llm.module import Thing`. No relative imports. Optional/slow imports inside methods or under `if typing.TYPE_CHECKING:`.
- **Naming**: No abbreviations (use `batch_size` not `bs`). Private members get a single `_` prefix; never use `__`. Keep public interfaces lean.
- **Types**: Always type-hint public interfaces. Use modern syntax (`X | Y`, `list[T]` not `List[T]`, PEP 695 generics like `class X[T: Bound]:` instead of `typing.TypeVar`).
- **Assert**: Use the `Assert` namespace from `fast_llm.utils` for contract checks (`Assert.eq`, `Assert.geq`, `Assert.incl`, `Assert.custom`, etc.) — error messages auto-format with actual values. Bare `assert` is reserved for internal state-validity invariants (`assert self._is_setup`).
- **Exceptions**: Raise stdlib exceptions for runtime errors (`ValueError`, `RuntimeError`, `NotImplementedError`). Custom exception classes are rare — only `ValidationError`, `NestedValidationError`, `FieldTypeError` in `config.py`.
- **Logging**: `logger = logging.getLogger(__name__)` per module. Use `info`/`warning`/`error`; `logger.debug` is not used in this codebase. For rank-aware logging, use `log_main_rank` from `fast_llm.engine.config_utils.run`.
- **`zip(...)`**: Always pass `strict=True` — length mismatch is a bug worth catching.
- **Conditionals**: Avoid double negations — prefer `b if x else a` over `a if not x else b`.
- **Paths**: Use `pathlib.Path`, not `os.path`.
- **Python version**: 3.12+.

## PR review focus

When reviewing PRs (`/review` reads this section as part of its checklist):

- **Consistency** with the rest of the codebase. New code should match neighbor patterns; flag arbitrary divergences from existing conventions.
- **Necessity**: flag anything that doesn't pull its weight — dead code, unused parameters, abstractions that aren't reused, comments that restate obvious code.
- **Simplification**: actively look for non-trivial simplifications and refactoring opportunities, not just style nitpicks.
- **Correctness**: edge cases, off-by-ones, error handling, race conditions. Read the code; don't trust comments.
- **Test coverage**: new code paths should have tests; modified behavior should have updated tests. Untested control flow is a flag.
- **Diff discipline**: don't substitute commit messages or `git --stat` for reading the code. On a follow-up review after fixes, read `git diff <last-reviewed-sha>..HEAD` in full before claiming an item is "verified" or "fixed". Never put ✅ on items whose diff you haven't read.
