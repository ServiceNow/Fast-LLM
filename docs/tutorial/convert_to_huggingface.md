# Converting Fast-LLM Models to Hugging Face Format

Now that we have trained a Mistral model, the natural next step is to try it for inference or benchmarks.
Fast-LLM does not support such task (at least for the time being),
but instead supports conversion to [Huggingface transformers](https://github.com/huggingface/transformers) models,
which are themselves compatible with a large variety of tools.

This article guides you through the conversion process for a Mistral-7B checkpoint (export)
generated during training as described in [the previous tutorial](launch_training.md).
This checkpoint may be found at `$EXP_BASE_DIR/export/$ITERATION/`.
Allow some time for the first checkpoint to be generated.


## Convert a Mistral-7B checkpoint

We convert the checkpoint with Fast-LLM's
[conversion script](https://github.com/ServiceNow/Fast-LLM/blob/main/tools/convert_model.py),
and we specify the input and output locations and formats:

```bash
python3 -m tools.convert_model \
    --input_type distributed \
    --output_type huggingface \
    --input_path $EXP_BASE_DIR/export/$ITERATION/ \
    --output_path $CONVERTED_DIR \
    --model_type mistral
```

<!--- TODO: What Tokenizer? --->

!!! warning "Don't Forget the Tokenizer"

    Make sure to add a tokenizer file and its configuration to the output directory, since `convert_model.py` does not include these files in the conversion.


<!--- TODO: What Tokenizer? --->

You can then load and use the converted model
[as you would with any Transformers model](https://huggingface.co/docs/transformers/index).
For example:
```python
import torch
from transformers import AutoModelForCausalLM

import transformers

model = AutoModelForCausalLM.from_pretrained(converted_dir).to(device="cuda")
x = torch.randint(0, 32000, (1, 1024))
y = model(x)
```
