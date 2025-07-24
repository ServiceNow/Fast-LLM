import pathlib

import pytest
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.models.ssm.config import LLambaHuggingfaceCheckpointFormat
from fast_llm.models.ssm.model import HybridSSMModel


@pytest.mark.skip("Disabled due to cartesia_pytorch installation issue")
@pytest.mark.slow
def test_load_from_llamba_checkpoint():
    """
    Test to check whether the of Fast-LLM and Huggingface checkpoint loading for Llamba-1B produce the same results.
    """
    import cartesia_pytorch.Llamba.llamba

    vocab_size = 128256  # from https://huggingface.co/cartesia-ai/Llamba-1B/blob/main/config.json
    batch_size = 2
    seq_length = 32

    path = pathlib.Path("/mnt/checkpoints_fml/pretrained_models/Llamba-1B")
    format = LLambaHuggingfaceCheckpointFormat

    x = torch.randint(0, vocab_size, (batch_size, seq_length), device="cuda")

    hf_model = cartesia_pytorch.Llamba.llamba.LMHeadModel.from_pretrained(path, strict=True).to("cuda")
    parameter_sum_hf = sum(p.detach().sum().cpu().item() for p in hf_model.parameters())
    hf_logits = hf_model(x)["logits"].cpu()
    del hf_model
    torch.cuda.empty_cache()

    # Create checkpoint load config
    checkpoint_config = CheckpointLoadConfig(path=path, format=format, model_weights=True, optimizer_state=False)
    # Initialize model
    model = HybridSSMModel.from_pretrained(checkpoint_config)
    param_sum = 0
    for stage in model.stages:
        for fsdp in stage.fsdps:
            if hasattr(fsdp, "_weight_shard"):
                param_sum += torch.sum(fsdp._weight_shard).item()
    assert torch.abs(torch.tensor(param_sum) - parameter_sum_hf) < 1e-1

    # model = GPTModel.from_pretrained(checkpoint_config)
    assert model.config.base_model.vocab_size == vocab_size
    schedule_config = ScheduleConfig()
    with NoAutoValidate():
        batch_config = GPTBatchConfig(micro_batch_size=batch_size, sequence_length=seq_length)
    batch_config.setup(DistributedConfig.from_dict({}))
    batch_config.validate()
    schedule_runner = ScheduleRunner(
        config=schedule_config,
        multi_stage=model,
        distributed_config=model.distributed.config,
    )
    schedule = Schedule(
        multi_stage=model,
        batch_config=batch_config,
        schedule_config=schedule_config,
        distributed_config=model.distributed.config,
        phase=PhaseType.inference,
    )
    schedule_runner.setup(model.distributed, optimizer=None)

    common_kwargs = {
        TransformerKwargs.sequence_first: True,
        TransformerKwargs.grad_output: False,
    }
    input_data = [(x, common_kwargs)]

    schedule_runner.run_step(iter([input_data]), schedule, iteration=0, return_metrics=True, preprocessed=True)

    logits = input_data[0][1]["logits"].cpu()
    assert torch.allclose(logits, hf_logits, atol=1e-2)
