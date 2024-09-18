# Load Mistral-7B

## Download pretrained weights

Since we are interested in extending the pretraining of Mistral-7B, the first step is to obtain the pretrained weights.
We do so by downloading them from the [Huggingface Hub](https://huggingface.co/mistralai/Mistral-7B-v0.1).
This requires:

- Git lfs (`git lfs install`).
- An account for the Huggingface Hub, together with an [access token](https://huggingface.co/docs/hub/security-tokens).
- Permission to use [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1), obtained by accepting the terms and conditions.

Then, clone the repository to download the weights (use the access token as password).
```bash
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 $PRETRAINED_CHECKPOINT_PATH
```


## Load the model in Fast-LLM

Fast-LLM may load the model architecture and pretrained weights of supported Huggingface models directly at the beginning of training.
To do so, we simply specify the pretrained checkpoint format and location,
which overrides the model architecture with Mistral-7B.
```bash
export ARCHITECTURE_ARGS_MISTRAL_PRETRAINED="\
--pretrained_checkpoint_type=huggingface \
--pretrained_checkpoint_path=$PRETRAINED_MISTRAL_PATH \
"
```

To obtain the full model configuration, we also need to set the non-architecture parameters,
which are not imported during conversion.

```bash
export MODEL_ARGS_MISTRAL_PRETRAINED="\
$ARCHITECTURE_ARGS_MISTRAL_PRETRAINED \
--window_size=4096 \
"
```

!!! warning

    Make sure to check which model parameters are part of the architecture and which ones are not,
    and set all required non-architecture parameters explicitly.

!!! warning

    Make sure the downloaded checkpoint is accessible to every worker, and adjust the path as needed.


## (Optional) Train from scratch

If we want to train a Mistral-7B model from scratch, we may still load the architecture from the Huggingface repo:
```bash
export ARCHITECTURE_ARGS_MISTRAL_FROM_SCRATCH="\
--pretrained_checkpoint_type=huggingface \
--pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH \
--load_pretrained_weights=0 \
"
```

Alternatively, we may specify the architecture explicitly, which makes it easier to adjust the parameters.
```bash
export ARCHITECTURE_ARGS_MISTRAL="\
--num_layers=32 \
--hidden_size=4096 \
--vocab_size=32000 \
--num_attention_heads=32 \
--head_groups=8 \
--add_linear_biases=0 \
--ffn_hidden_size=14336 \
--kv_channels=128 \
--use_rotary_embeddings=1 \
--rotary_embedding_scale=-9.210340371976184 \
--gated=1 \
--activation_type=silu \
--normalization_type=rms_norm \
--tie_word_embeddings=0 \
"
```

Please refer to the trainer config for additional extended pretraining options.


## (Optional) Train Mixtral-8x7B

<!--- TODO: Move to separate file? --->

We may train Mixtral-8x7B instead, which simply requires pointing to a different checkpoint:

```bash
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1Mixtral-8x7B-v0.1 $PRETRAINED_CHECKPOINT_PATH
```
Other than a small memory optimization, this tutorial can be run as-is with Mixtral-8x7B.
The architecture is a slight vatiation of Mistral-7B:
```bash
export ARCHITECTURE_ARGS_MIXTRAL="\
$ARCHITECTURE_ARGS_MISTRAL \
--num_experts=8 \
--num_experts_per_token=2 \
"
```
