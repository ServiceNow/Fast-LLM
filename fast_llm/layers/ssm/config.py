from fast_llm.config import Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.layers.common.config import NormalizationConfig
from fast_llm.layers.transformer.config import TransformerArchitectureConfig
from fast_llm.tensor import TensorSpace
from fast_llm.utils import Assert


class SSMDimNames:
    d_model = "D_model"
    d_state = "D_state"
    d_conv = "D_conv"  # dimention of the conv1d input in mamba layers
    d_inner = "D_inner"
    dt_rank = "D_rank"
    d_inner_proj = "D_inner_proj"
    d_x_proj = "D_x_proj"
    headdim = "D_headdim"  # dimention of the mamba2 head
    d_conv_kernel = "D_conv_kernel"  # kernel size of the conv1d in mamba layers
    n_qk_heads = "N_qk_heads"
    n_v_heads = "N_v_heads"


@config_class()
class MambaConfig(TransformerArchitectureConfig, BaseModelConfig):
    """Configuration for a Structured State Space Model (SSM) layer."""

    dt_init_floor: float = Field(
        default=1e-4,
        desc="Minimum value for initializing dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    # Layer parameters
    add_bias_linear: bool = Field(
        default=False,
        desc="Whether to use bias in linear transformations",
        hint=FieldHint.core,
    )

    # Normalization
    normalization: NormalizationConfig = FieldUpdate(default_factory=NormalizationConfig)

    # Performance optimization
    use_fast_path: bool = Field(
        default=True,
        desc="Whether to use optimized CUDA kernels when available",
        hint=FieldHint.performance,
    )

    use_module_layernorm: bool = Field(
        default=False,
        desc="use_module_layernorm",
        hint=FieldHint.optional,
    )

    rms_norm: bool = Field(
        default=False,
        desc="rms_norm",
        hint=FieldHint.optional,
    )

    debug_ssm: bool = Field(
        default=False,
        desc="debug_ssm",
        hint=FieldHint.optional,
    )

    fused_add_norm: bool = Field(
        default=False,
        desc="fused_add_norm",
        hint=FieldHint.optional,
    )

    layernorm_epsilon: float = Field(
        default=1e-5,
        desc="layernorm_epsilon",
        hint=FieldHint.optional,
    )

    dt_min: float = Field(
        default=0.001,
        desc="Minimum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    dt_max: float = Field(
        default=0.1,
        desc="Maximum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    dt_init_floor: float = Field(
        default=1e-4,
        desc="Minimum value for initializing dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    residual_in_fp32: bool = Field(
        default=False,
        desc="residual_in_fp32",
        hint=FieldHint.optional,
    )
    expansion_factor: int = Field(
        default=2,
        desc="Expansion factor for Mamba blocks.",
        hint=FieldHint.core,
    )
    state_size: int = Field(
        default=16,
        desc="State size for Mamba blocks.",
        hint=FieldHint.core,
    )
    conv_kernel_dimension: int = Field(
        default=4,
        desc="Conv kernel dimension for Mamba blocks.",
        hint=FieldHint.core,
    )
    rms_norm: bool = Field(
        default=True,
        desc="Use RMS normalization for Mamba blocks.",
        hint=FieldHint.core,
    )

    residual_in_fp32: bool = Field(
        default=True,
        desc="Use residual in fp32 for Mamba blocks.",
        hint=FieldHint.core,
    )
    fused_add_norm: bool = Field(
        default=False,
        desc="Use fused add norm for Mamba blocks.",
        hint=FieldHint.core,
    )
    layernorm_epsilon: float = Field(
        default=1e-5,
        desc="Epsilon for layer normalization for Mamba blocks.",
        hint=FieldHint.core,
    )

    dt_rank: str | int = Field(
        default="auto",
        desc="Rank of the Δ projection matrix. If 'auto', set to ceil(hidden_size/16)",
        hint=FieldHint.core,
    )

    # Mamba2 parameters
    use_mamba2: bool = Field(
        default=False,
        desc="Use Mamba2 blocks.",
        hint=FieldHint.core,
    )

    chunk_size: int = Field(
        default=256,
        desc="Chunk size for Mamba2 blocks.",
        hint=FieldHint.core,
    )

    n_qk_heads: int = Field(
        default=32,
        desc="Number of QK heads for Mamba2 blocks.",
        hint=FieldHint.core,
    )

    n_v_heads: int = Field(
        default=32,
        desc="Number of V heads for Mamba2 blocks.",
        hint=FieldHint.core,
    )

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        pass

    def _validate(self) -> None:
        """Validate configuration parameters."""

        super()._validate()

        # Validate SSM-specific parameters
        Assert.gt(self.state_size, 0)
        Assert.gt(self.expansion_factor, 0)
        Assert.gt(self.conv_kernel_dimension, 0)
        Assert.gt(self.dt_min, 0)
        Assert.gt(self.dt_max, 0)
        Assert.gt(self.dt_init_floor, 0)
        Assert.geq(self.dt_max, self.dt_min)

        if isinstance(self.dt_rank, int):
            Assert.gt(self.dt_rank, 0)
