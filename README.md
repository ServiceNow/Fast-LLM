<div align="center" style="margin-bottom: 1em;">

<img width=50% src="docs/assets/images/logo.png" alt="Fast-LLM Logo"></img>

[![Tests][tests-badge]][tests]
[![Documentation][docs-badge]][docs]
[![License][license-badge]][license]

*Accelerating your LLM training to full speed*

Made with ❤️ by [ServiceNow Research][servicenow-research]

</div>

## Overview

Fast-LLM is a new open-source library for training large language models. It's design focuses on speed, scalability, flexibility, and ease of use. Fast-LLM is built on top of [PyTorch](https://pytorch.org/) and [Triton](https://triton-lang.org) to provide a state-of-the-art training experience.

## Why Fast-LLM?

1. 🚀 **Fast-LLM is Blazing Fast**:
    - ⚡️ Optimized kernel efficiency and reduced overheads.
    - 🔋 Better memory usage.
    - ⏱⏳ Reduced training time and cost.
  
2. 📈 **Fast-LLM is Highly Scalable**:
    - 📡 Distributed training across multiple GPUs and nodes using 3D parallelism (DP+TP+PP).
    - 🔄 Supports sequence length parallelism.
    - 🧠 ZeRO-1, ZeRO-2, and ZeRO-3 offloading for memory efficiency.
    - 🎛️ Support for mixed precision training.
    - 🏋️‍♂️ Large batch training and gradient accumulation support.

3. 🎨 **Fast-LLM is Incredibly Flexible**:
    - 🤖 Compatible with all common language model architectures in a unified class.
    - ⚡ Efficient dropless Mixture-of-Experts (MoE) support.
    - 🧩 Customizable for language model architectures, data loaders, loss functions, and optimizers.
    - 🤗 Compatible with [Hugging Face Transformers](https://huggingface.co/transformers/).

4. 🎯 **Fast-LLM is Super Easy to Use**:
    - 📦 Pre-built Docker images for fast deployment.
    - 📝 Simple YAML configuration for hassle-free setup.
    - 💻 Command-line interface for seamless launches.
    - 📊 Detailed logging and real-time monitoring features.
    - 📚 Comprehensive documentation and helpful tutorials.

5. 🌐 **Fast-LLM is Truly Open Source**:
    - ⚖️ Apache 2.0 License.
    - 💻 Fully developed on GitHub with a public roadmap and transparent issue tracking.
    - 🤝 Contributions and collaboration are always welcome!

## Usage

## Next Steps

  **Want to learn more?** Check out our [documentation](https://servicenow.github.io/Fast-LLM) for more information on how to use Fast-LLM.

🔨 **We welcome contributions to Fast-LLM!** Have a look at our [contribution guidelines](CONTRIBUTING.md).

🐞 **Something doesn't work?** Open an [issue](https://github.com/ServiceNow/Fast-LLM/issues)!

## License

Fast-LLM is licensed by ServiceNow, Inc. under the Apache 2.0 License. See [LICENSE](LICENSE) for more information.

## Vulnerability Reporting

If you find a security vulnerability in Fast-LLM, please report it to us at [psirt-oss@servicenow.com](mailto:psirt-oss@servicenow.com) as soon as possible. Please refer to our [security policy](SECURITY.md) for more information.

[tests-badge]: https://github.com/ServiceNow/Fast-LLM/actions/workflows/pythonci.yml/badge.svg
[tests]: https://github.com/ServiceNow/Fast-LLM/actions/workflows/pythonci.yml
[docs-badge]: https://github.com/ServiceNow/Fast-LLM/actions/workflows/docs_cd.yml/badge.svg
[docs]: https://servicenow.github.io/Fast-LLM
[license-badge]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license]: LICENSE
[servicenow-research]: https://www.servicenow.com/research/
