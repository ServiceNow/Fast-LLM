# PR Title Guidelines ✏️

Since we squash commits when merging pull requests, the PR title will become the commit message for the squashed commit. To ensure a clear and consistent project history, follow these guidelines for naming your PR:

1. **Use a concise yet descriptive title**: The title should summarize the key change or feature introduced. Avoid vague titles like "Fix bug" or "Update code."
2. **Start with a keyword**: Use keywords to categorize the type of change. For example:
   - **feat:** for new features (e.g., `[feat] add support for mixed-precision training`)
   - **fix:** for bug fixes (e.g., `[fix] resolve memory leak during backpropagation`)
   - **perf:** for performance improvements (e.g., `[perf] optimize gradient accumulation step`)
   - **refactor:** for code refactoring (e.g., `[refactor] clean up data loader module`)
   - **docs:** for documentation changes (e.g., `[docs] update contributing guidelines`)
   - **build:** for changes to the build process or dependencies (e.g., `[build] bump PyTorch version`)
3. **Reference the issue number (if applicable)**: If the PR is related to a specific issue, include the issue number in the title (e.g., `[fix] resolve #123 memory leak in training loop`).
