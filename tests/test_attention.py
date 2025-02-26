import unittest.mock
from fast_llm.layers.transformer.attention import Attention
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.config_utils.tensor_space import TensorSpace


def test_decide_window_size():
    attention = unittest.mock.Mock(spec=Attention)
    attention._decide_window_size = Attention._decide_window_size.__get__(attention)  # Attach real method

    # Arrange - Case 1: window_size is returned (layer_index >= max_window_layers)
    attention._config = TransformerConfig(window_size=512, max_window_layers=2)
    attention._layer_index = 2
    assert attention._decide_window_size() == 512

    # Arrange - Case 2: window_size is None (layer_index < max_window_layers)
    attention._config = TransformerConfig(window_size=512, max_window_layers=2)
    attention._layer_index = 1
    assert attention._decide_window_size() is None

    # Arrange - Case 3: max_window_layers is None (always return window_size)
    attention._config = TransformerConfig(window_size=512, max_window_layers=None)
    assert attention._decide_window_size() == 512


def test_attention_constructor():
    transformer_conf = TransformerConfig(
        num_layers=2, 
        num_attention_heads=2,
        hidden_size=16,
    )
    distributed_config = DistributedConfig()
    tensor_space = TensorSpace(distributed_config=distributed_config)
    transformer_conf.setup_tensor_space(tensor_space)

    Attention(transformer_conf, tensor_space, 1)

