# How to Release Fast-LLM

This document provides a step-by-step guide for creating a new release of Fast-LLM. Follow these instructions to ensure a smooth and consistent release process.

## Release Policy

1.  **Responsibility:** Only the maintainer is authorized to create and publish releases. This ensures consistency, quality, and accountability in the release process.

2.  **Collaboration:** Contributors with write access can propose changes and prepare the repository for a release (e.g., ensuring tests pass, updating documentation). However, the final tagging and publishing of a release are the maintainer's responsibility.

3.  **Versioning:** Follow [Semantic Versioning](https://semver.org/) guidelines (e.g., `MAJOR.MINOR.PATCH`).

## Release Process

### 1. Update Version

1.  Open the `__init__.py` file and update the `__version__` string to the new version number.
2.  Update the `version` field in `setup.cfg` to match.

    ```python
    # __init__.py
    __version__ = "0.2.0"  # Update this to the new version.
    ```

    ```properties
    # setup.cfg
    version = "0.2.0"  # Update this to the new version.
    ```

3.  Commit the changes:

    ```bash
    git add __init__.py setup.cfg
    git commit -m "Bump version to 0.2.0"
    ```

### 2. Create a Git Tag

1.  Create an annotated tag for the release:

    ```bash
    git tag -a v0.2.0 -m "Release version 0.2.0"
    ```

2.  Push the tag to GitHub:

    ```bash
    git push origin v0.2.0
    ```

### 3. Draft a Release on GitHub

1.  Go to the repository on GitHub and click on the **Releases** tab.
2.  Click **Draft a new release**.
3.  Under **Tag version**, select the tag you just pushed (`v0.2.0`).
4.  Use the **Generate release notes** button to automatically create release notes based on merged pull requests and commits.
5.  Customize the release notes as needed:

    -   Highlight new features, bug fixes, and other changes.
    -   Group pull requests by categories such as "Features", "Bug Fixes", and "Documentation".

6.  Click **Publish release**.

### 4. Verify CI/CD Pipeline

1.  Confirm that all Continuous Integration (CI) workflows have completed successfully (e.g., tests, builds).
2.  If applicable, verify that the release artifacts (i.e., docker images) are available and accessible.

### 5. Post-Release Checklist

1.  Announce the new release to the community.
2.  Verify that the updated documentation (if any) is live.
3.  Prepare for the next development cycle by:

    -   Creating a new branch for the next version.
    -   Updating the `__version__` string and `setup.cfg` to a development version (e.g., `0.2.1-dev`).

    ```python
    # __init__.py
    __version__ = "0.2.1-dev"
    ```

    ```properties
    # setup.cfg
    version = "0.2.1-dev"
    ```

4.  Push changes to the repository:

   ```bash
   git add __init__.py pyproject.toml
   git commit -m "Start development on version 0.2.1"
   git push origin main
   ```
