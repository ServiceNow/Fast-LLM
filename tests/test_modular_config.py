from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.layers.ssm.config import SSMConfig
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.models.hybrid.config import HybridBaseModelConfig, MambaBlockConfig, TransformerBlockConfig
from fast_llm.models.hybrid.model import HybridBaseModel

config = HybridBaseModelConfig(
    blocks={
        "transformer_block": TransformerBlockConfig(
            transformer=TransformerConfig(
                hidden_size=4096,
                num_attention_heads=32,
                num_layers=10,
            ),
        ),
        "mamba_block": MambaBlockConfig(
            ssm=SSMConfig(
                state_size=16,
            ),
        ),
        "mamba2_block": MambaBlockConfig(
            ssm=SSMConfig(
                state_size=16,
            ),
        ),
    },
    hybrid_block_layout=["mamba_block", "mamba2_block", "mamba_block"],
)

distributed_config = DistributedConfig(
    tensor_parallel=1,
    pipeline_parallel=1,
    world_size=1,
)

# Create model
model = HybridBaseModel(config, distributed_config)
