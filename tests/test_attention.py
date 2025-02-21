import unittest.mock
from fast_llm.layers.transformer.attention import Attention
from fast_llm.layers.transformer.config import TransformerConfig


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
