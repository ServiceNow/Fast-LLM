# How to Release Fast-LLM: A Step-by-Step Guide

This document walks you through the process of creating a new release of Fast-LLM. We follow these steps to keep releasing smooth, consistent, and hassle-free.

## Release Policy

1.  **Who's in Charge?** Only the maintainer is authorized to create and publish releases. This ensures consistency, quality, and accountability.

2.  **Teamwork Makes the Dream Work:** Contributors with write access can propose changes and prep the repository for a release (steps 1 and 2 below in the "Release Process" section). But tagging and publishing the release? That's the maintainer's job.

3.  **Versioning Made Simple:** Fast-LLM sticks to [Semantic Versioning](https://semver.org/) (aka semver). Here's the gist:
    -   **MAJOR versions** (like `1.0.0`) are for big, stable, feature-complete milestones. Since we're still in pre-1.0 territory, we don't have these yet.
    -   **MINOR versions** (e.g., `0.2.0`) introduce new features and may include breaking changes, as we are in the pre-1.0 phase of development. While we strive for backward compatibility where feasible, breaking changes are acceptable until we reach 1.0.0. MINOR releases are the main focus of our current development efforts. They're tied to [milestones](https://github.com/ServiceNow/Fast-LLM/milestones) and are released on a regular schedule.
    -   **PATCH versions** (e.g., `0.2.1`) squash bugs and include small, critical fixes without introducing new functionality. These releases are based on stable `main` commits and are the recommended choice for production-like use cases and important experiments. While we encourage internal and adventurous users to test `main`, PATCH releases ensure stability for users who need reliability.

4.  **Milestones are for MINOR Releases:** Each [milestone](https://github.com/ServiceNow/Fast-LLM/milestones) corresponds to a MINOR version (`0.2.0`, `0.3.0`, etc.) and includes all issues and pull requests targeted for that release. Milestones have due dates and are used to track progress toward the next MINOR release. PATCH releases? Handled as individual issues or small groups of issues.

5.  **All Roads Lead to `main`:** Active development happens on the `main` branch, which may include breaking changes. For production experiments or stability-critical use cases, use the latest PATCH or MINOR release. Internally, we encourage testing `main` to identify issues early, but important experiments should always use tagged releases to ensure reproducibility and compatibility.

## Release Process

### 1. Get Ready to Release

Before tagging anything, make sure the repository is in tip-top shape:

1.  Close or defer all issues in the current milestone (where applicable).
2.  Verify that all targeted pull requests are merged.
3.  Double-check the repo:

    -   All tests should pass.
    -   The documentation should be up to date.
    -   Pull requests should have appropriate labels for release notes.

4.  Decide if unresolved bugs need fixing before the release.

### 2. Update the Version

1.  Update the version in `__init__.py` and `setup.cfg`:

    ```python
    # __init__.py
    __version__ = "0.2.0"  # Update this to the new version.
    ```

    ```properties
    # setup.cfg
    version = "0.2.0"  # Update this to the new version.
    ```

2.  Commit the version bump:

    ```bash
    git add __init__.py setup.cfg
    git commit -m "Bump version to 0.2.0"
    ```

### 3. Tag It

1.  Create a new Git tag:

    ```bash
    git tag -a v0.2.0 -m "Release version 0.2.0"
    ```

2.  Push the tag to GitHub:

    ```bash
    git push origin v0.2.0
    ```

### 4. Draft a Release on GitHub

1.  Head to the [Releases section](https://github.com/ServiceNow/Fast-LLM/releases) in the Fast-LLM GitHub repository.
2.  Click [Create a new release](https://github.com/ServiceNow/Fast-LLM/releases/new).
3.  Under **Choose a tag**, select the tag you just pushed (e.g., `v0.2.0`).
4.  Use GitHub's [automatic release note generation](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes) feature by clicking **Generate release notes** to create release notes based on merged pull requests and commits since the last release.
5.  Customize the release notes as needed by highlighting key changes and features.
6.  Activate the **Create a discussion for this release** option to allow users to ask questions and provide feedback.
7.  Click **Publish release** to make the release public.

### 5. Check the CI/CD Pipeline

1.  Confirm all CI workflows for the tagged version are green (including tests, docker builds, documentation).
2.  Verify that the release artifacts (e.g., Docker images) are available.
3.  Ensure updated documentation is live.

### 6. Post-Release Checklist

1.  **Spread the Word:** Announce the release across Fast-LLM's communication channels, which includes the GitHub Discussions forum and the discussion thread for the release created in step 4.
2.  **After the release is before the release:** Prep for the next version:

    -   Update the `__version__` string and `setup.cfg` to reflect the next development version (e.g., `0.2.1-dev`):

        ```python
        # __init__.py
        __version__ = "0.2.1-dev"
        ```

        ```properties
        # setup.cfg
        version = "0.2.1-dev"
        ```

    -   Commit and push the changes:

        ```bash
        git add __init__.py setup.cfg
        git commit -m "Start development on version 0.2.1"
        git push origin main
        ```

3.  Update milestones:

    -   Close the milestone for the release (e.g., `0.2.0`).
    -   Create a new milestone for the next MINOR release (e.g., `0.3.0`).
