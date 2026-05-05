---
description: Fine-grained self-review of the current branch — naming, comments, dead code
---

Run a fine-grained self-review of the current branch's diff against `main`. This is **pass 2 of 2**, run after `/review-coarse` and after structural fixes have landed. Assume the code's shape is settled: do **not** propose structural rewrites, new abstractions, or large refactors — bring those up only if they're so important they justify reopening pass 1.

## How to read the diff

Read `git diff main...HEAD` in full before commenting. Do not substitute commit messages or `git --stat` for reading the code.

On a follow-up review after fixes have been pushed, run `git diff <last-reviewed-sha>..HEAD` and read the full output. Never mark an item ✅ whose diff you haven't read.

## Focus

- **Naming**: unclear, abbreviated, or inconsistent identifiers. Project convention: no abbreviations (`batch_size` not `bs`); private members use single `_` prefix, never `__`.
- **Comments**: comments that restate the code, reference the current task/PR/issue (those belong in the PR description), or document obvious behavior. Default is no comment unless the *why* is non-obvious — a hidden constraint, subtle invariant, or workaround for a specific bug.
- **Dead code**: unused parameters, unreached branches, leftover debug prints, vestigial helpers, abstractions with a single caller, defensive checks (validation, fallbacks, `try/except`, `if x is None: raise`) for situations that can't happen or inputs that should be trusted at internal boundaries, comments restating what well-named identifiers already say.
- **Redundancy**: duplicated logic that could collapse, repeated literals that could be constants.
- **Style nits**: typing — modern syntax (`X | Y`, `list[T]`, PEP 695 generics) and presence on public interfaces (flag missing type hints); imports (third-party fully qualified, first-party `from fast_llm.module import Thing`); `Assert` namespace vs bare `assert`; `zip(..., strict=True)`; `pathlib.Path` over `os.path`; stdlib exceptions for runtime errors (no new custom exception classes); `logger.info`/`warning`/`error` only (no `.debug`), `log_main_rank` for rank-aware logging; no double negations.

## Output

Each numbered item must stand on its own as a concrete, actionable finding — a specific change to make or a clear problem to fix, phrased so `fix N` is a complete instruction. Keep items concise by default; add context (rationale, alternatives considered) only when it's needed for the contributor to act. Surrounding sections (overview, summary, framing) are fine; this constraint applies only to the numbered list.

When an item refers to specific code, include a path from the repo root and the relevant line number(s), in the form `path/to/file.py:42` or `path/to/file.py:42-58`.

Number every item (1, 2, 3...) so items can be referenced by number (`fix 2 and 4`, `ignore 5`). Don't use unnumbered bullets where ordinals would make items addressable.
