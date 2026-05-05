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
- **Test coverage**: new code paths should have tests; modified behavior should have updated tests. Untested control flow is a flag.
- **Performance**: a new feature must add **no measurable overhead when unused**. On the disabled path, flag: new kernel launches, new GPU sync points, a slower GPU code path, new CPU work inside training hot loops (forward/backward, schedule loop, per-step dataloader path), and any added cost that scales with model size, sequence length, batch size, or step count. Trivial additions outside hot loops — a config-flag branch, an extra attribute read, a one-shot validation in `__init__` — are fine. Prefer gating new behavior behind a config flag that short-circuits cheaply when off.
- **Security**: untrusted input reaching `eval`/`exec`/shell/SQL/path-construction without sanitization, secrets in code or logs, deserialization of untrusted data, weak crypto, and any new attack surface introduced by the change.
- **Scope**: features, abstractions, validation, fallbacks, or backwards-compat shims beyond what the task requires.

## Output

Number every review item (1, 2, 3...) so items can be referenced by number (`fix 2 and 4`, `ignore 5`). Don't use unnumbered bullets where ordinals would make items addressable.
