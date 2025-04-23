from fast_llm.config import Field, FieldHint, FieldUpdate, check_field, config_class
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig, BaseModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.config import NormalizationArchitectureConfig, NormalizationConfig
from fast_llm.utils import Assert


class SSMDimNames:
    model_dim = "model_dim"  # Model dimension (D)
    state_dim = "state_dim"  # State dimension (N)
    conv_dim = "conv_dim"  # Dimension of the conv1d input in mamba layers
    inner_dim = "inner_dim"  # Inner dimension after expansion
    dt_rank = "dt_rank"  # Rank of Δ
    inner_proj_mamba = "inner_proj_mamba"  # Inner projection dimension for mamba
    inner_proj_mamba2 = "inner_proj_mamba2"  # Inner projection dimension for mamba2
    x_proj_dim = "x_proj_dim"  # X projection dimension
    head_dim = "head_dim"  # Dimension of the mamba2 head (P)
    conv_kernel_size = "conv_kernel_size"  # Kernel size of the conv1d in mamba layers
    qk_heads = "qk_heads"  # Number of QK heads
    v_heads = "v_heads"  # Number of V heads


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

    dt_rank: int = Field(
        default=None,
        desc="Rank of the Δ projection matrix. If 'None', will be set to ceil(hidden_size/16)",
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

    activation_type: ActivationType = Field(
        default=None,
        desc="The MLP intermediate activation type. Default: SiLU for gated MLP, GeLU otherwise.",
        hint=FieldHint.core,
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.activation_type is None:
                self.activation_type = ActivationType.silu
            if self.dt_rank is None:
                self.dt_rank = -1  # set to -1, it will be overwrittem in ssm validation

        super()._validate()


@config_class()
class SSMConfig(SSMArchitectureConfig, BaseModelConfig):
    """Configuration for a Structured State Space Model (SSM) layer."""

    normalization: NormalizationConfig = FieldUpdate(default_factory=NormalizationConfig)

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

    def _validate(self) -> None:
        """Validate configuration parameters."""

        super()._validate()
        Assert.geq(self.dt_max, self.dt_min)
