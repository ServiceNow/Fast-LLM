# Contributing to Fast-LLM 🚀

Thank you for your interest in contributing to Fast-LLM. We appreciate your support in helping accelerate LLM training to full speed! This document outlines the guidelines for contributing to Fast-LLM, including reporting issues, submitting changes, and setting up the development environment.

If you just want to ask a question, please [open a discussion](https://github.com/ServiceNow/Fast-LLM/discussions) on our GitHub page.

## Getting Started

To contribute to Fast-LLM, follow these steps to set up your environment:

1. **Set Up the Development Environment**: Fast-LLM is based on [PyTorch](https://pytorch.org/) and [Triton](https://triton-lang.org/) and development depends on them. See our [setup guide](https://servicenow.github.io/Fast-LLM/development/setup) for instructions on setting up both the development environment and dependencies.
3. **Learn Our Best Practices**: Familiarize yourself with our [development best practices](https://servicenow.github.io/Fast-LLM/development/dev-practices/), including code style, pre-commit hooks, and testing strategies.
4. **Launch Fast-LLM Locally or with Docker**: Instructions can be found in the [launching section](https://servicenow.github.io/Fast-LLM/development/launching).

## How to Report a Bug 🐞

Encountered a problem? [Open an issue](https://github.com/ServiceNow/Fast-LLM/issues/new/choose) and select "Bug report." Provide as much detail as possible, including:
- Steps to reproduce the issue.
- Expected and actual behavior.
- Screenshots, log files, or error messages (if applicable).
- Your environment setup, including OS, Docker version, and relevant configurations.

If you're familiar with the codebase, adding a failing unit test that demonstrates the issue can be helpful, but it's not required.

## Proposing Changes

Before starting any work, please [open an issue](https://github.com/ServiceNow/Fast-LLM/issues) to discuss your proposal, especially if it involves significant changes or adding a new dependency. Once your proposal has been approved, follow these steps:

1. **Fork the Repository**: [Fork Fast-LLM](https://github.com/ServiceNow/Fast-LLM/fork) to your own GitHub account.
2. **Clone Your Fork Locally**: Use `git clone` to bring the code to your local machine.
3. **Create a New Branch**: Name your branch descriptively, e.g., `feature/new-feature` or `fix/bug-name`.
4. **Make Changes**: Modify the code and add or update tests.
5. **Commit Your Changes**: Use descriptive commit messages. Follow our [commit message guidelines](https://servicenow.github.io/Fast-LLM/development/commit-guidelines/).
6. **Push to Your Fork**: Push the branch to your GitHub fork.
7. **Open a Pull Request**: [Submit a pull request](https://github.com/ServiceNow/Fast-LLM/compare) to the `main` branch. Reference the original issue number and include a summary of your changes.

### Guidelines for a Successful Pull Request

To ensure your pull request gets reviewed promptly, please:
- **Follow our coding standards**: Adhere to our [development best practices](https://servicenow.github.io/Fast-LLM/development/dev-practices/).
- **Write tests**: Include unit tests for new features or bug fixes.
- **Comment non-trivial code**: Make the code understandable for others.
- **Do not expose sensitive data**: Avoid including private or proprietary information in your code or commit messages.
- **Review the [PR template](https://github.com/ServiceNow/Fast-LLM/blob/main/.github/pull_request_template.md)**: Complete the checklist before submission.

## Seeking Help or Clarification

If you need help, feel free to reach out via:
- **GitHub Discussions**: [Open a discussion](https://github.com/ServiceNow/Fast-LLM/discussions) if you have a question or need guidance.
- **Project Maintainers**: Mention a maintainer in an issue or pull request if you need a review or help. 

## Contributors

We recognize and appreciate all the contributors who have made Fast-LLM what it is today. Join our contributors' list by making your first contribution!

To learn more about the team and maintainers, visit our [About page](https://servicenow.github.io/Fast-LLM/about-us/).
