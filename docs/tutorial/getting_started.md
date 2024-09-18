# Getting Started

<!--- TODO: Remove the ServiceNow-specific content. --->

## Build the image

!!! warning

    This guide is not yet working.

The preferred way to run [Fast-LLM](https://github.com/ServiceNow/Fast-LLM) is through a docker image built with the provided Dockerfile.
For example, from a terminal running on a GPU node:

```bash
git clone git@github.com:ServiceNow/Fast-LLM.git
cd Fast-LLM
docker build -t my_fast_llm_image .
docker run --rm -it --gpus all --net=host --ipc=host my_fast_llm_image bash
```

## First examples

All training runs are launched throught the entry point [pretrain_fast_llm.py](https://github.com/ServiceNow/Fast-LLM/blob/main/pretrain_fast_llm.py).
We can run a minimalistic training example with:
```bash
python3 pretrain_fast_llm.py --train_iters=100 --batch_size=32 --dataset_source=random
```
This will launch a short single-GPU training from scratch of a 180 M parameter model on a randomly generated dataset.

To run distributed training, we run our training script through [torchrun](https://pytorch.org/docs/stable/elastic/run.html),
the PyTorch distributed launcher. For example, on 8 GPUs:
```bash
torchrun --nproc-per-node=8 pretrain_fast_llm.py --train_iters=100 --batch_size=32 --dataset_source=random
```
Note that by default, Fast-LLM parallelizes over samples (data-parallel), so the number of GPUs should divide the batch size.

Multi-node training also uses torchrun, and requires the same command to be run on each node,
with the additional specification of a rendez-vous endpoint, i.e., the address of one of the nodes.
For example, on four nodes:
```bash
torchrun --nproc-per-node=8 --nnodes=4 --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR pretrain_fast_llm.py --train_iters=100 --batch_size=32 --dataset_source=random
```

See the [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html) for more details.
Note that if you are using cloud or managed hardware, there Now tutorial](servicenow.md)
may be a simpler automated method to launch multi-node jobs.
Please refer to your provider for more details.
The ServiceNow-specific method may be found in the [Service

## More on training arguments

<!--- TODO: Document arguments --->

The training script supports hundreds of arguments, though most of them are optional and/or have sensible defaults.
We already saw three arguments above, and we will see many important ones in this tutorial.

At the beginning of training, Fast-LLM displays a list of arguments and their values:
```
------------------------ arguments ------------------------
  activation_type ................................. gelu
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  add_linear_biases ............................... True
  attention_dropout ............................... 0.0
  batch_size ...................................... 1
  [...]
-------------------- end of arguments ---------------------
```
All of these arguments can be set as arguments of `pretrain_fast_llm.py`, in the form `--[name]=[value]`,
provided the values have the expected data type, and in some case satisfy extra constraints.
For example, we may enable attention dropout with `--attention_dropout=0.1`.
Note that booleans are set as integers (ex. `--add_linear_biases=0` to disable biases),
and that `None` cannot be represented.
Please refer to each parameter's definition for more details.
