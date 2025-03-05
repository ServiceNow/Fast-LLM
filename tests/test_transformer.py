from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.common.config import NormalizationType
from fast_llm.models.gpt.config import GPTBaseModelConfig
from fast_llm.models.gpt.model import GPTBaseModel
from fast_llm.utils import Assert


def test_variable_window_size():
    model = GPTBaseModel(
        GPTBaseModelConfig.from_dict(
            {
                "layers": {
                    "default": {"window_size": 1024, "num_layers": 8, "normalization": {"type": "rms_norm"}},
                    "layers": [
                        {
                            # Layers 5, 6 and 7
                            "layer_ranges": [{"begin": 5, "end": None}],
                            "updates": {"window_size": None, "normalization": {"epsilon": 1}},
                        },
                        {
                            # Layers 0, 3 and 5, but 5 already covered above so excluded.
                            "layer_ranges": [{"begin": 0, "end": 1}, {"begin": 3, "end": 6, "step": 2}],
                            "updates": {"window_size": 512, "ffn_hidden_size": 64},
                        },
                    ],
                }
            }
        ),
        DistributedConfig(training_dtype=DataType.bfloat16),
    )
    Assert.eq(
        [layer._config.window_size for layer in model.layers[1:-1]], [512, 1024, 1024, 512, 1024, None, None, None]
    )
    Assert.eq([layer._config.normalization.type for layer in model.layers[1:-1]], [NormalizationType.rms_norm] * 8)
    Assert.eq([layer._config.normalization.epsilon for layer in model.layers[1:-1]], [1e-5] * 5 + [1] * 3)
    Assert.eq(
        [layer._config.ffn_hidden_size for layer in model.layers[1:-1]], [64, 4096, 4096, 64, 4096, 4096, 4096, 4096]
    )
    # Non-architecture parameters (`window_size`) need to be ignored when converting to architecture config.
    # (See `TransformerLayerRangeArchitectureConfig.setup`.)
    model.config.get_architecture()
