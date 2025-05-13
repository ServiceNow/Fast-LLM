---
title: How to Generate with a Fast-LLM Model
---

Fast-LLM models support `generate` and `forward` operations through Hugging Face–compatible wrappers.

⚠️ Limitations:

- No support for `cache`, `past_key_values`, `labels`, `attention` outputs, or `inputs_embeds`
- `position_ids` are ignored and reconstructed from the attention mask
- **model-parallel** and **sequence-data-parallel** generation is **not** supported

---

### 🔧 Generating Text from a Fast-LLM Model

Below is a step-by-step example of how to generate text using a Fast-LLM model checkpoint from Hugging Face Hub.

```python
# Import dependencies
import huggingface_hub
from transformers import AutoTokenizer
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.models.gpt.config import LlamaGPTHuggingfaceCheckpointFormat
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM

# Specify model and configuration
model = "HuggingFaceTB/SmolLM2-135M-Instruct"
checkpoint_format = LlamaGPTHuggingfaceCheckpointFormat
max_new_tokens = 50

# Download model checkpoint from the Hugging Face Hub to a local directory
model_path = huggingface_hub.snapshot_download(repo_id=model, local_dir="/tmp")

# Load tokenizer from the downloaded model
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Optional: updates to Fast-LLM config before loading the model
updates = {
    ("base_model", "transformer", "use_flash_attention"): True,
    ("distributed", "training_dtype"): "bf16"
}

# Load the model from the checkpoint with the given configuration
model = HuggingfaceGPTModelForCausalLM.from_pretrained(
    CheckpointLoadConfig(
        path=model_path,
        format=checkpoint_format,
        model_weights=True,
    ),
    updates,
)

# Example input messages formatted for chat-style generation
messages = [
    {"role": "user", "content": "What is gravity?"},
    {"role": "user", "content": "Who is the president of EU?"},
]

# Convert messages into model input format using chat template
input_text = [tokenizer.apply_chat_template([el], tokenize=False) for el in messages]

# Prepare tokenized input for the model
tokenizer.padding_side = "left"  # Important for correct padding
inputs = tokenizer(input_text, padding="longest", return_tensors="pt").to("cuda")

# Generate text using the model
outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)

# Decode and display outputs
outputs = [tokenizer.decode(el, skip_special_tokens=True) for el in outputs]

print("--------------------------------------------------------------------")
for el in outputs:
    print(el)
    print("--------------------------------------------------------------------")
```
