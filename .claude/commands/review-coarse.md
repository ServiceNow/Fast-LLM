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
- **Scope**: features, abstractions, validation, fallbacks, or backwards-compat shims beyond what the task requires.

## Output

Number every review item (1, 2, 3...) so items can be referenced by number (`fix 2 and 4`, `ignore 5`). Don't use unnumbered bullets where ordinals would make items addressable.
