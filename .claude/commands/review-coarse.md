---
description: Coarse self-review of the current branch — structure, correctness, tests
---

Run a coarse self-review of the current branch's diff against `main`. This is **pass 1 of 2**: focus on issues whose fix could change the code's shape. Do **not** flag naming, comments, formatting, or other surface nits — those belong to `/review-fine` (pass 2), and flagging them now risks rework once structural changes land.

## How to read the diff

Read `git diff main...HEAD` in full before commenting. Do not substitute commit messages or `git --stat` for reading the code.

On a follow-up review after fixes have been pushed, run `git diff <last-reviewed-sha>..HEAD` and read the full output. Never mark an item ✅ whose diff you haven't read.

## Focus

- **Correctness**: edge cases, off-by-ones, error handling, race conditions, broken invariants. Read the code; don't trust comments.
- **Structure & consistency**: does new code match neighbor patterns and existing abstractions? Flag arbitrary divergences from existing conventions and parallel implementations of things the codebase already provides. New features should extend existing abstractions, not fork them for a specific use case.
- **Simplification**: actively look for non-trivial refactoring opportunities — places where the change could be smaller, an abstraction could be reused, or a new abstraction isn't earning its keep.
- **Necessity**: whole-unit deadweight — new modules, classes, abstractions, or code paths that don't pull their weight; config options that don't toggle meaningful behavior.
- **Test coverage**: new code paths should have tests; modified behavior should have updated tests. Untested control flow is a flag.
- **Performance**:
    - **General**: regressions on hot paths, accidental quadratic behavior, redundant work in inner loops, unnecessary allocations or host↔device transfers, GPU sync points, missed batching/fusion opportunities.
    - **Fast-LLM specific**: a new feature must add **no measurable overhead when unused**. On the disabled path, flag any new kernel launch, any new GPU sync, a slower GPU code path, new CPU work inside training hot loops (forward/backward, schedule loop, per-step dataloader path), and any added cost that scales with model size, sequence length, batch size, or step count. Trivial additions outside hot loops — a config-flag branch, an extra attribute read, a one-shot validation in `__init__` — are fine. Prefer gating new behavior behind a config flag that short-circuits cheaply when off.
- **Security**: untrusted input reaching `eval`/`exec`/shell/SQL/path-construction without sanitization, secrets in code or logs, deserialization of untrusted data, weak crypto, and any new attack surface introduced by the change.
- **Scope**: features, abstractions, validation, fallbacks, or backwards-compat shims beyond what the task requires.

## Output

Each numbered item must stand on its own as a concrete, actionable finding — a specific change to make or a clear problem to fix, phrased so `fix N` is a complete instruction. Prefer a single recommendation; only present alternatives ("either X or Y") when both are genuinely viable and you can't justify picking one. Keep items concise by default; add context (rationale, alternatives considered) only when it's needed for the contributor to act.

When an item refers to specific code, include a path from the repo root and the relevant line number(s), in the form `path/to/file.py:42` or `path/to/file.py:42-58`.

Number every item (1, 2, 3...). Insert a literal blank line in the source between consecutive numbered items — `2.` must not begin on the line right after item `1.`'s last line — so the rendered output has visible vertical spacing and items don't run together. Don't use unnumbered bullets where ordinals would make items addressable.

Surrounding sections (overview, summary, framing) are fine. Put non-actionable observations — maintenance hazards, meta-comments, things you noticed but can't propose a concrete action for — in a `## Notes` section *after* the numbered list, not inside it.
