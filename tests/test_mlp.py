from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.block.config import BlockConfig
from fast_llm.layers.block.mlp.mixture_of_experts import MixtureOfExpertMLP
from fast_llm.layers.block.mlp.mlp import MLP


def test_mlp_constructor():
    transformer_conf = BlockConfig(
        num_layers=2,
        num_attention_heads=2,
        hidden_size=16,
    )
    distributed_config = DistributedConfig()
    tensor_space = TensorSpace(distributed_config=distributed_config)
    transformer_conf.setup_tensor_space(tensor_space)

    MLP(transformer_conf, tensor_space, 0, "name")


def test_moe_mlp_constructor():
    transformer_conf = BlockConfig(
        num_layers=2, num_attention_heads=2, hidden_size=16, num_experts=2, add_linear_biases=False
    )
    distributed_config = DistributedConfig()
    tensor_space = TensorSpace(distributed_config=distributed_config)
    transformer_conf.setup_tensor_space(tensor_space)

    MixtureOfExpertMLP(transformer_conf, tensor_space, 0, "name")
