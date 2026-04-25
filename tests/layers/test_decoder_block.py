import pytest
import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.decoder.block import DecoderBlock
from fast_llm.layers.decoder.config import DecoderBlockConfig
from tests.utils.utils import get_stage


@pytest.mark.slow
@pytest.mark.parametrize(
    "norm_flags",
    [
        ("mixer_only", True, False),
        ("mlp_only", False, True),
        ("both", True, True),
    ],
)
def test_post_norm_gradients(norm_flags):
    """
    Verify that post_mixer_normalization and post_mlp_normalization weight gradients
    are correctly accumulated after a decoder block backward pass,
    confirming the norms participate in the computation graph.
    """
    _, use_post_mixer_norm, use_post_mlp_norm = norm_flags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads, head_groups, head_size = 4, 2, 32
    hidden_size = heads * head_size  # 128

    distributed_config = DistributedConfig(compute_dtype="float32", use_cuda=torch.cuda.is_available())
    hidden_dim = TensorDim("hidden_size", hidden_size)

    config_dict = {
        "mixer": {
            "head_size": head_size,
            "heads": heads,
            "head_groups": head_groups,
            "rotary": {"type": "none"},
            "implementation": "backup",
        },
        "mlp": {},
        "normalization": {"type": "rms_norm"},
    }
    if use_post_mixer_norm:
        config_dict["post_mixer_normalization"] = {"type": "rms_norm"}
    if use_post_mlp_norm:
        config_dict["post_mlp_normalization"] = {"type": "rms_norm"}

    block: DecoderBlock = DecoderBlockConfig._from_dict(config_dict).get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None
    )
    distributed = Distributed(distributed_config)
    get_stage([block], distributed)

    lengths = [10, 15]
    num_tokens = sum(lengths)
    (model_input,) = LanguageModelBatch(
        tokens=torch.empty(num_tokens, dtype=torch.int64, device=device), lengths=lengths
    ).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            return_document_index=True,
        )
    )
    kwargs = model_input.to_kwargs()
    block.preprocess(kwargs)

    input_ = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device=device, requires_grad=True)
    output = block(input_, kwargs)
    output.sum().backward()

    def _check_grad_nonzero(param, name):
        # triton/fused backward writes to grad_buffer and unsets param_grad_is_zero;
        # torch fallback writes to param.grad and leaves param_grad_is_zero=True.
        if not getattr(param, "param_grad_is_zero", True):
            grad = param.grad_buffer
        else:
            grad = param.grad
        assert grad is not None, f"{name} weight grad is None"
        assert grad.abs().sum() > 0, f"{name} weight grad is all-zero"

    if use_post_mixer_norm:
        _check_grad_nonzero(block.post_mixer_normalization.weight, "post_mixer_normalization")

    if use_post_mlp_norm:
        _check_grad_nonzero(block.post_mlp_normalization.weight, "post_mlp_normalization")
