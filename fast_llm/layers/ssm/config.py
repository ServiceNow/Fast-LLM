from fast_llm.config import Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig
from fast_llm.layers.common.config import NormalizationArchitectureConfig, NormalizationConfig
from fast_llm.tensor import TensorSpace
from fast_llm.utils import Assert


class SSMDimNames:
    d_model = "model_dimension_D"
    d_state = "state_dimension_N"
    d_conv = "size_of_conv1d_input"  # dimention of the conv1d input in mamba layers
    d_inner = "inner_dimension_after_expansion"
    dt_rank = "rank_of_Δ"
    d_inner_proj_m = "inner_projection_dimension_mamba"
    d_inner_proj_m2 = "inner_projection_dimension_mamba2"
    d_x_proj = "x_projection_dimension"
    headdim = "head_dimension_P"  # dimention of the mamba2 head
    d_conv_kernel = "1d_conv_kernel_size"  # kernel size of the conv1d in mamba layers
    n_qk_heads = "number_of_qk_heads"
    n_v_heads = "number_of_v_heads"


@config_class()
class SSMArchitectureConfig(BaseModelArchitectureConfig):
    _abstract = False

    # Normalization
    normalization: NormalizationArchitectureConfig = Field(
        default_factory=NormalizationArchitectureConfig,
        desc="Configuration for the normalization layers architecture.",
        hint=FieldHint.core,
    )

    expansion_factor: int = Field(
        default=2, desc="Expansion factor for Mamba blocks.", hint=FieldHint.core, valid=check_field(Assert.gt, 0)
    )

    state_size: int = Field(
        default=16,
        desc="State size for Mamba blocks.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    conv_kernel_dimension: int = Field(
        default=4,
        desc="Conv kernel dimension for Mamba blocks.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    # Layer parameters
    add_bias_linear: bool = Field(
        default=False,
        desc="Whether to use bias in SSM layers",
        hint=FieldHint.core,
    )

    dt_rank: str | int = Field(
        default="auto",
        desc="Rank of the Δ projection matrix. If 'auto', set to ceil(hidden_size/16)",
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

    activation: str = Field(
        default="silu",
        desc="Activation function for Mamba2 blocks.",
        hint=FieldHint.core,
    )


@config_class()
class SSMLayerConfig(SSMArchitectureConfig):
    """Configuration for a Structured State Space Model (SSM) layer."""

    normalization: NormalizationConfig = FieldUpdate(default_factory=NormalizationConfig)
    # Performance optimization
    use_fast_path: bool = Field(
        default=True,
        desc="Whether to use optimized CUDA kernels when available",
        hint=FieldHint.performance,
    )

    debug_ssm: bool = Field(
        default=False,
        desc="debug_ssm",
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

    def setup_tensor_space(self, tensor_space: TensorSpace) -> None:
        pass

    def _validate(self) -> None:
        """Validate configuration parameters."""

        super()._validate()
        Assert.geq(self.dt_max, self.dt_min)

        if isinstance(self.dt_rank, int):
            Assert.gt(self.dt_rank, 0)
