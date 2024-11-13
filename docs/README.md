# Fast-LLM Documentation Sources

This folder contains the source files for the Fast-LLM documentation. The contents here are used to generate the rendered documentation, which is automatically updated and published whenever changes are pushed to the `main` branch.

## 📚 Access the Rendered Documentation

To view the complete, rendered documentation, please visit the [Fast-LLM Documentation Site](https://servicenow.github.io/Fast-LLM).

## Building and Serving the Documentation

To build and preview the documentation locally, follow these simple steps:

1.  **Install the necessary dependencies:**

    ```bash
    pip install --no-build-isolation -e ".[DOCS]"
    ```

    You also need to install `libcairo` for image processing on your system. Follow <https://squidfunk.github.io/mkdocs-material/plugins/requirements/image-processing/> for more details.

2.  **Build the documentation:**

    ```bash
    mkdocs build
    ```

    This will generate the static documentation files in a `site/` folder.

3.  **Serve the documentation locally (with auto-reload):**

    ```bash
    mkdocs serve
    ```

    The documentation site will be served locally at [http://127.0.0.1:8000](http://127.0.0.1:8000), and any changes made to the source files will automatically trigger a rebuild.

## Contributing to the Documentation

If you'd like to contribute to the Fast-LLM documentation, feel free to edit these source files and submit a pull request. The changes will be reflected on the rendered documentation site after they are merged into the `main` branch.

Your contributions could be as simple as helping to correct typos and spelling errors, improving existing content to provide more details on how to approach a tricky step for novice users, or even to add new content that describes functionality with limited or no detailed coverage anywhere else. No matter how small, we value all contributions from the Fast-LLM community.
