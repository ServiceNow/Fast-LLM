from fast_llm.layers.transformer.mlp import MLP
from fast_llm.layers.transformer.mixture_of_experts import MixtureOfExpertMLP
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.config_utils.tensor_space import TensorSpace


def test_mlp_constructor():
    transformer_conf = TransformerConfig(
        num_layers=2,
        num_attention_heads=2,
        hidden_size=16,
    )
    distributed_config = DistributedConfig()
    tensor_space = TensorSpace(distributed_config=distributed_config)
    transformer_conf.setup_tensor_space(tensor_space)

    MLP(transformer_conf, tensor_space, "name")


def test_moe_mlp_constructor():
    transformer_conf = TransformerConfig(
        num_layers=2,
        num_attention_heads=2,
        hidden_size=16,
        num_experts=2,
        add_linear_biases=False
    )
    distributed_config = DistributedConfig()
    tensor_space = TensorSpace(distributed_config=distributed_config)
    transformer_conf.setup_tensor_space(tensor_space)

    MixtureOfExpertMLP(transformer_conf, tensor_space, "name")
