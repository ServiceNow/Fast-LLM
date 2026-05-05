import dataclasses
import functools

import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.common.normalization.normalization import NoNormalization
from fast_llm.layers.decoder.block import DecoderBlock
from fast_llm.layers.decoder.config import DecoderBlockConfig
from tests.utils.utils import get_stage

_NUM_TOKENS = 16
_HIDDEN_SIZE = 64
_HEADS = 4
_KV_HEADS = 2
_HEAD_SIZE = 16
_INTERMEDIATE_SIZE = 128


@dataclasses.dataclass
class PostNormTestConfig:
    name: str
    post_mixer_norm: bool = False
    post_mlp_norm: bool = False
    output_scale: float | None = None
    # Per-position pre-norm override; when set, replaces the block's default `normalization`
    # at that position (None means "inherit default"; `{"type": "none"}` disables the norm).
    pre_mixer_normalization: dict | None = None
    pre_mlp_normalization: dict | None = None

    def get_block_config(self) -> DecoderBlockConfig:
        config_dict: dict = {
            "mixer": {
                "heads": _HEADS,
                "head_groups": _KV_HEADS,
                "head_size": _HEAD_SIZE,
                "add_linear_biases": False,
                "implementation": "backup",
            },
            "mlp": {
                "intermediate_size": _INTERMEDIATE_SIZE,
                "add_linear_biases": False,
            },
            "normalization": {"type": "rms_norm"},
        }
        if self.post_mixer_norm:
            config_dict["post_mixer_normalization"] = {"type": "rms_norm"}
        if self.post_mlp_norm:
            config_dict["post_mlp_normalization"] = {"type": "rms_norm"}
        if self.output_scale is not None:
            config_dict["output_scale"] = {"enabled": True}
        if self.pre_mixer_normalization is not None:
            config_dict["pre_mixer_normalization"] = self.pre_mixer_normalization
        if self.pre_mlp_normalization is not None:
            config_dict["pre_mlp_normalization"] = self.pre_mlp_normalization
        return DecoderBlockConfig.from_dict(config_dict)

    @functools.cached_property
    def threshold(self) -> float:
        return 1e-5

    def expected_output(self, block: DecoderBlock, input_: torch.Tensor, kwargs: dict) -> torch.Tensor:
        # Block-assembly test. The mixer and MLP are treated as black boxes (covered by
        # `test_attention` and `test_mlp` respectively); norms/residual/output_scale are computed
        # via `torch.rms_norm` so the assembly under test does not appear in its own reference.
        def _rms_norm(x: torch.Tensor, norm_module) -> torch.Tensor:
            if isinstance(norm_module, NoNormalization):
                return x
            return torch.rms_norm(x, x.shape[-1:], norm_module.weight, 1e-5)

        with torch.no_grad():
            norm1_out = _rms_norm(input_, block.norm_1)
            mixer_hidden, mixer_bias = block.mixer(norm1_out, kwargs)
            if block.post_mixer_norm is not None:
                mixer_hidden = _rms_norm(mixer_hidden, block.post_mixer_norm)
            if mixer_bias is not None:
                mixer_hidden = mixer_hidden + mixer_bias
            after_mixer = input_ + mixer_hidden

            norm2_out = _rms_norm(after_mixer, block.norm_2)
            mlp_hidden, mlp_bias = block.mlp(norm2_out, kwargs)
            if block.post_mlp_norm is not None:
                mlp_hidden = _rms_norm(mlp_hidden, block.post_mlp_norm)
            if mlp_bias is not None:
                mlp_hidden = mlp_hidden + mlp_bias
            output = after_mixer + mlp_hidden
            if self.output_scale is not None:
                output = output * self.output_scale
            return output


_base_post_norm_cases = [
    ("no_post_norms", {}),
    ("post_mixer_norm", {"post_mixer_norm": True}),
    ("post_mlp_norm", {"post_mlp_norm": True}),
    ("both_post_norms", {"post_mixer_norm": True, "post_mlp_norm": True}),
    ("output_scale", {"output_scale": 2.5}),
    # `{"type": "none"}` disables the position-specific pre-norm. Gemma 4's MoE block path uses
    # this to skip the pre-MLP norm (the routed branch owns its own pre/post norms).
    ("pre_mixer_norm_disabled", {"pre_mixer_normalization": {"type": "none"}}),
    ("pre_mlp_norm_disabled", {"pre_mlp_normalization": {"type": "none"}}),
    (
        "pre_norms_disabled",
        {
            "pre_mixer_normalization": {"type": "none"},
            "pre_mlp_normalization": {"type": "none"},
        },
    ),
]

_post_norm_test_configs = [PostNormTestConfig(name=name, **kwargs) for name, kwargs in _base_post_norm_cases]


@pytest.mark.parametrize(
    "test_config",
    [pytest.param(c, id=c.name) for c in _post_norm_test_configs],
)
def test_post_norms(test_config: PostNormTestConfig):
    distributed_config = DistributedConfig(use_cuda=torch.cuda.is_available())
    distributed = Distributed(distributed_config)
    hidden_dim = TensorDim("hidden", _HIDDEN_SIZE)
    block: DecoderBlock = test_config.get_block_config().get_layer(
        distributed_config, hidden_dim, lr_scale=None, peft=None
    )
    get_stage([block], distributed)
    block.eval()

    device = distributed.device
    if test_config.output_scale is not None:
        with torch.no_grad():
            block.output_scale.fill_(test_config.output_scale)
    input_ = torch.randn(_NUM_TOKENS, _HIDDEN_SIZE, device=device)

    token_dim = TensorDim("token", _NUM_TOKENS)
    kwargs = {
        AttentionKwargs.sequence_k_dim: TensorDim("sequence_k", _NUM_TOKENS),
        AttentionKwargs.token_dim: token_dim,
        AttentionKwargs.hidden_token_dim: token_dim,
        AttentionKwargs.key_value_token_dim: token_dim,
        AttentionKwargs.sequence_length: _NUM_TOKENS,
        AttentionKwargs.document_index_k: torch.zeros(_NUM_TOKENS, dtype=torch.int64, device=device),
        AttentionKwargs.document_index_q: torch.zeros(_NUM_TOKENS, dtype=torch.int64, device=device),
        AttentionKwargs.device: device,
    }
    block.preprocess(kwargs)

    with torch.no_grad():
        output = block(input_, kwargs)

    expected = test_config.expected_output(block, input_, kwargs)
    torch.testing.assert_close(output, expected, rtol=test_config.threshold, atol=test_config.threshold)
