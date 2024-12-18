---
title: Contributing
---

Thank you for your interest in contributing to Fast-LLM! We're thrilled to have you here, and your support is invaluable in helping us accelerate LLM training to full speed. This guide will walk you through the steps to contribute, from reporting issues to submitting changes and setting up your development environment.

If you have questions or want to start a discussion, feel free to [open a discussion](https://github.com/ServiceNow/Fast-LLM/discussions) on our GitHub page.

## 🚀 Getting Started

To get started with contributing to Fast-LLM, follow these steps to set up your environment:

1.  **Learn Our Development Practices**: Get familiar with our [development best practices](https://servicenow.github.io/Fast-LLM/developers/dev-practices), which cover development setup, testing, and benchmarking.
2.  **Read the Style Guide**: Follow our [style guide](https://servicenow.github.io/Fast-LLM/developers/style-guide) to maintain consistency in code style, documentation, and commit messages.

## 🐞 How to Report a Bug

Found a bug? Let's squash it together! [Open an issue](https://github.com/ServiceNow/Fast-LLM/issues/new/choose) and select "Bug report." Please include as much information as possible:

-   Steps to reproduce the issue.
-   What you expected to happen versus what actually happened.
-   Logs, Fast-LLM configuration, and error messages.
-   Details about your environment setup (e.g., CUDA hardware, PyTorch version, CUDA version).

If you're familiar with the codebase, consider adding a failing unit test to demonstrate the problem (optional, but helpful!).

## 🛠️ Proposing Changes

Before diving into code, [open an issue](https://github.com/ServiceNow/Fast-LLM/issues) to discuss your proposal. This is especially important if you're planning significant changes or adding new dependencies. Once your idea is approved, follow these steps:

1.  **Fork the Repository**: [Fork Fast-LLM](https://github.com/ServiceNow/Fast-LLM/fork) to your own GitHub account.
2.  **Clone Your Fork Locally**: Use `git clone` to bring the code to your local machine.
3.  **Create a New Branch**: Name your branch descriptively, such as `fix/training-memory-leak` or `feature/rope-scaling`.
4.  **Make Your Changes**: Work your magic! Don't forget to add or update tests, benchmarks, or configurations as needed.
5.  **Push to Your Fork**: Push the branch to your GitHub fork.
6.  **Open a Pull Request**: [Submit a pull request](https://github.com/ServiceNow/Fast-LLM/compare) to the `main` branch. Reference the original issue number and provide a brief summary of your changes.

## 🏆 Guidelines for a Successful Pull Request

Here are some tips to ensure your pull request gets reviewed and merged promptly:

-   **Follow our coding standards**: Stick to our [style guide and conventions](https://servicenow.github.io/Fast-LLM/developers/style-guide) to keep the code clean and consistent.
-   **Write tests**: Verify your changes with unit tests for new features or bug fixes.
-   **Test on GPUs and real-world workloads**: Since Fast-LLM is all about training large language models, make sure your changes work smoothly in GPU environments and on typical training setups.
-   **Run benchmarks and performance tests**: Make sure your changes don't slow things down. If there's any impact on performance, provide benchmark results to back it up.
-   **Avoid introducing new issues**: Check that there are no new runtime warnings, type checker errors, linting problems, or unhandled edge cases.
-   **Comment non-trivial code**: Make your code easy to understand for others.
-   **Keep sensitive data out**: Make sure your code or commit messages don't expose private or proprietary information.
-   **Use a clear and descriptive title**: The PR title should summarize the key change or feature introduced. Avoid vague titles like "Fix bug" or "Update code." Start with a keyword like `[feat]`, `[fix]`, `[docs]`, etc. to categorize the change. Reference the issue number if applicable (e.g., `[fix] resolve #123 memory leak in training loop`). This title will become the commit message for the squashed merge.
-   **Use the [PR template](https://github.com/ServiceNow/Fast-LLM/blob/main/.github/PULL_REQUEST_TEMPLATE.md)**: Complete the checklist to make sure everything is in order before hitting submit.

## 🆘 Seeking Help or Clarification

If you're unsure about something or need help, you've got options:

-   **GitHub Discussions**: [Start a discussion](https://github.com/ServiceNow/Fast-LLM/discussions) if you need advice or just want to chat.
-   **Project Maintainers**: Mention a maintainer in an issue or pull request if you need a review or guidance.

## 🌟 Contributors

We're grateful for all the awesome contributors who help make Fast-LLM better. Join our contributors' list and make your first contribution!

To learn more about the team and maintainers, visit our [About page](https://servicenow.github.io/Fast-LLM/about-us).
