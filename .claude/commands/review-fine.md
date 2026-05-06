---
description: Fine-grained self-review of the current branch — naming, comments, dead code
---

<!-- Maintenance: the "How to read the diff" and "Output" sections must stay byte-identical with .claude/commands/review-coarse.md. Slash commands have no include mechanism; keep them in sync by hand. -->

Run a fine-grained self-review of the current branch's diff against `main`. This is **pass 2 of 2**, run after `/review-coarse` and after structural fixes have landed. Assume the code's shape is settled: do **not** propose structural rewrites, new abstractions, or large refactors — bring those up only if they're so important they justify reopening pass 1.

## How to read the diff

Read `git diff main...HEAD` in full before commenting. Do not substitute commit messages or `git --stat` for reading the code.

On a follow-up review after fixes have been pushed, run `git diff <last-reviewed-sha>..HEAD` and read the full output. Never mark an item ✅ whose diff you haven't read.

## Focus

- **Naming**: unclear, abbreviated, or inconsistent identifiers. Project convention: no abbreviations (`batch_size` not `bs`); private members use single `_` prefix, never `__`.
- **Comments**: comments that restate the code, reference the current task/PR/issue (those belong in the PR description), or document obvious behavior. Default is no comment unless the *why* is non-obvious — a hidden constraint, subtle invariant, or workaround for a specific bug.
- **Dead code**: unused parameters, unreached branches, leftover debug prints, vestigial helpers, abstractions with a single caller, defensive checks (validation, fallbacks, `try/except`, `if x is None: raise`) for situations that can't happen or inputs that should be trusted at internal boundaries.
- **Redundancy**: duplicated logic that could collapse, repeated literals that could be constants.
- **Style nits**: typing — modern syntax (`X | Y`, `list[T]`, PEP 695 generics) and presence on public interfaces (flag missing type hints); imports (third-party fully qualified, first-party `from fast_llm.module import Thing`); `Assert` namespace vs bare `assert`; `zip(..., strict=True)`; `pathlib.Path` over `os.path`; stdlib exceptions for runtime errors (no new custom exception classes); `logger.info`/`warning`/`error` only (no `.debug`), `log_main_rank` for rank-aware logging; no double negations.

## Output

Each numbered item must stand on its own as a concrete, actionable finding — a specific change to make or a clear problem to fix, phrased so `fix N` is a complete instruction. Prefer a single recommendation; only present alternatives ("either X or Y") when both are genuinely viable and you can't justify picking one. Keep items concise by default; add context (rationale, alternatives considered) only when it's needed for the contributor to act.

When an item refers to specific code, include a path from the repo root and the relevant line number(s), in the form `path/to/file.py:42` or `path/to/file.py:42-58`. Verify line numbers by reading the relevant file section with the `Read` tool — diff hunk headers (`@@`) are easy to miscount and must not be used as the sole source.

Format each item as a **standalone paragraph beginning with a bolded number**, e.g. `**1.** <finding>...`, *not* as a Markdown ordered list (`1. ...\n2. ...`) — terminal renderers collapse ordered lists to a tight layout regardless of source blank lines, but standalone paragraphs render with normal vertical spacing. Insert a blank line between items. Don't use unnumbered bullets where ordinals would make items addressable.

Surrounding sections (overview, summary, framing) are fine. Put non-actionable observations — maintenance hazards, meta-comments, things you noticed but can't propose a concrete action for — in a `## Notes` section *after* the numbered list, not inside it.
