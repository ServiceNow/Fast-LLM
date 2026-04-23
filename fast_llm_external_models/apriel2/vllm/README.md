# vLLM Support for Apriel2

## Usage

Register Apriel2 before creating the LLM:

```python
from fast_llm_external_models.apriel2.vllm import register
from vllm import LLM

register()

llm = LLM(model="path/to/apriel2/checkpoint")
```

## Entry Point (Alternative)

Add to your `pyproject.toml` to auto-register on vLLM import:

```toml
[project.entry-points."vllm.plugins"]
apriel2 = "fast_llm_external_models.apriel2.vllm:register"
```
