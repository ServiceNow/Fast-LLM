# Evaluations

Fast-LLM allows you to perform various evaluations during training or as a separate evaluation step. In both cases, you need to use your training config with `training.evaluators` specified.

For evaluators used during training, both `interval` and `offset` must be specified. Then, start training as usual with:

`fast-llm train gpt --config path/to/training/config.yaml`

To perform evaluation as a separate step, use the same training config. Depending on the training progress, either the start model or the latest checkpoint will be loaded, and `interval` and `offset` will be ignored. To start evaluation:

`fast-llm evaluate gpt --config path/to/training/config.yaml`

## Currently Supported Evaluators

- `loss`
- `lm_eval`

## Loss Evaluator

To set up loss evaluation, specify a dataset to be used in the `data.datasets` section of the config. You must also define the loss evaluator in the `training.evaluators` config section. See example below.

```yaml
training:
  evaluations:
    stack_3b:
      interval: 10
      evaluator:
        type: loss
        iterations: 10
        dataset_name: stack_3b
    fineweb:
      evaluator:
        type: loss
        iterations: 10
        dataset_name: stack_3b
      interval: 10
data:
  datasets:
    stack_3b:
      type: memmap
      path: path/to/memmap/dataset
    fineweb:
      type: memmap
      path: path/to/memmap/dataset1
```

## Evaluation Harness (`lm_eval`) Evaluator

**Note:** Only data parallelism is currently supported for the `lm_eval` evaluator.

To run `lm_eval` evaluations, version `0.4.9` of `lm_eval` must be installed along with all dependencies required for your evaluation tasks.

The following environment variables may need to be set:

- `HF_HOME`: Path for Hugging Face data caching
- `WANDB_API_KEY_PATH`: Path to a file containing your Weights & Biases API key (if logging to W&B)
- `HUGGINGFACE_API_KEY_PATH`: Path to a file containing your Hugging Face hub token
- `NLTK_DATA`: Path to a directory that will contain downloaded NLTK packages (needed for some tasks)
- `HF_ALLOW_CODE_EVAL=1`: Required for some evaluation tasks

You may need to specify additional environment variables depending on the `lm_eval` tasks you want to run.

To specify an `lm_eval` task, the evaluator config includes the following fields:

### Model Config

The model instantiated for training is reused for evaluation, so you don't need to specify it separately. However, there are some parameters specific to `lm_eval`. See `fast_llm/engine/evaluation/config.EvaluatorLmEvalConfig` for details.

### CLI Parameters for `lm_eval`

All other parameters are specified as if you were calling the `lm_eval` CLI, using a list of strings. Some CLI parameters are ignored or restricted—specifically those related to model loading, W&B, batch sizes, and device setup, as these are managed by the rest of the Fast-LLM configuration.

Also, the tokenizer must be specified in `data.tokenizer`. If the tokenizer does not have a `bos_token`, it must be specified explicitly in `data.tokenizer.bos_token`. Although `lm_eval` does not use the `bos_token` directly, it is still required because the same tokenizer is used by other Fast-LLM components.

Below is an example of the config:

```yaml
training:
  evaluations:
    lm_eval_tasks1:
      interval: 10
      evaluator:
        type: lm_eval
        cli_args:
          - --tasks
          - gsm8k,xnli_en,wikitext,ifeval
          - --output_path
          - /path/to/lm_eval/output
data:
  tokenizer:
    path: path/to/the/tokenizer
```

It is also possible to run different tasks with different intervals and offsets—for example, to run slower or more comprehensive tasks less frequently.:

```yaml
training:
  evaluations:
    gsm8k:
      interval: 20
      evaluator:
        type: lm_eval
        cli_args:
          - --tasks
          - gsm8k
          - --output_path
          - /path/to/lm_eval/output
          - --limit
          - "64"
    ifeval:
      offset: 10
      interval: 40
      evaluator:
        type: lm_eval
        cli_args:
          - --tasks
          - ifeval
          - --output_path
          - /path/to/lm_eval/output
          - --limit
          - "32"
    faster_tasks:
      interval: 10
      evaluator:
        type: lm_eval
        cli_args:
          - --tasks
          - xnli_en,wikitext
          - --output_path
          - /path/to/lm_eval/output
data:
  tokenizer:
    path: path/to/the/tokenizer
```
