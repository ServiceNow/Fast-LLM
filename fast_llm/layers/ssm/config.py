import enum

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.config import LLMBlockConfig, NormalizationConfig
from fast_llm.utils import Assert


class SSMDimNames:
    model_dim = "model_dim"  # Model dimension (D)
    state_dim = "state_dim"  # State dimension (N)
    conv_dim = "conv_dim"  # Dimension of the conv1d input in mamba layers
    inner_dim = "inner_dim"  # Inner dimension after expansion
    dt_rank = "dt_rank"  # Rank of Δ
    inner_proj_mamba = "inner_proj_mamba"  # Inner projection dimension for mamba
    inner_proj_discrete_mamba2 = "inner_proj_discrete_mamba2"  # Inner projection dimension for discrete mamba2
    inner_proj_mamba2 = "inner_proj_mamba2"  # Inner projection dimension for mamba2
    x_proj_dim = "x_proj_dim"  # X projection dimension
    head_dim = "head_dim"  # Dimension of the mamba2 head (P)
    conv_kernel_size = "conv_kernel_size"  # Kernel size of the conv1d in mamba layers
    qk_heads = "qk_heads"  # Number of QK heads
    v_heads = "v_heads"  # Number of V heads

    # Mamba 2
    x_proj_dim_2 = "x_proj_dim"  # d_xb


class SSMBlockType(enum.StrEnum):
    """
    An enum for the available mamba types for the MLP layer.
    """

    mamba = "m"
    mamba2_discrete = "m2d"
    mamba2 = "m2"
    transformer = "t"


@config_class()
class SSMConfig(LLMBlockConfig):
    _abstract = False

    # Normalization
    normalization: NormalizationConfig = Field(
        desc="Configuration for the normalization layers architecture.",
        hint=FieldHint.architecture,
    )
    expansion_factor: int = Field(
        default=2,
        desc="Expansion factor for Mamba blocks.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    state_size: int = Field(
        default=16,
        desc="State size for Mamba blocks.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    conv_kernel_dimension: int = Field(
        default=4,
        desc="Conv kernel dimension for Mamba blocks.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    # Layer parameters
    add_bias_linear: bool = Field(
        default=False,
        desc="Whether to use bias in SSM layers",
        hint=FieldHint.architecture,
    )

    dt_rank: None | int = Field(
        default=None,
        desc="Rank of the Δ projection matrix. If 'None', will be set to ceil(hidden_size/16)",
        hint=FieldHint.architecture,
    )
    chunk_size: int = Field(
        default=256,
        desc="Chunk size for Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
    n_qk_heads: int = Field(
        default=32,
        desc="Number of QK heads for Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
    n_v_heads: int = Field(
        default=32,
        desc="Number of V heads for Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
    activation_type: ActivationType = Field(
        default=None,
        desc="The MLP intermediate activation type. Default: SiLU for gated MLP, GeLU otherwise.",
        hint=FieldHint.architecture,
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
    dt_init_floor: float = Field(
        default=1e-4,
        desc="Minimum value for initializing dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    d_inner: None | int = Field(
        default=None,
        desc="Inner dimension for Mamba2 blocks.",
        hint=FieldHint.core,
    )
    mamba_lr_scale: float | None = Field(
        default=None,
        desc="Learning rate scale for Mamba blocks.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )

    # Mamba 2
    repeat_kv_before_conv: bool = Field(
        default=True,
        desc="Whether to repeat the KV before the conv1d in Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
    d_xb: int = Field(
        default=None,
        desc="Dimension of the xB in Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
    dt_init: str = Field(
        default="random",
        desc="Initialization method for dt",
        hint=FieldHint.core,
    )
    dt_max: float = Field(
        default=0.1,
        desc="Maximum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    dt_min: float = Field(
        default=0.001,
        desc="Minimum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    dt_init_floor: float = Field(
        default=1e-4,
        desc="Minimum value for initializing dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    dt_scale: float = Field(
        default=1.0,
        desc="Scale for dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.activation_type is None:
                self.activation_type = ActivationType.silu
        super()._validate()
        Assert.geq(self.dt_max, self.dt_min)
