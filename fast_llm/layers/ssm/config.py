import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.config import LLMBlockConfig, NormalizationConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.tensor import Initializer


class SSMBlockType(enum.StrEnum):
    """
    An enum for the available mamba types for the MLP layer.
    """

    mamba = "m"
    mamba2_discrete = "m2d"
    mamba2 = "m2"
    transformer = "t"

    def get_mixer_class(self):
        if self == SSMBlockType.mamba:
            from fast_llm.layers.ssm.mamba_layer import MambaLayer

            return MambaLayer
        elif self == SSMBlockType.mamba2:
            from fast_llm.layers.ssm.mamba2 import Mamba2

            return Mamba2
        elif self == SSMBlockType.mamba2_discrete:
            from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2

            return DiscreteMamba2
        else:
            raise NotImplementedError(self)


class DTInitType(enum.StrEnum):
    constant = "constant"
    random = "random"

    def get_init_method(self, scale: float) -> "Initializer":
        from fast_llm.tensor import init_fill_, init_uniform_centered_

        return init_fill_(scale) if self == DTInitType.constant else init_uniform_centered_(scale)


@config_class()
class SSMConfig(LLMBlockConfig):
    _abstract = False

    # Normalization
    normalization: NormalizationConfig = Field(
        desc="Configuration for the normalization layers architecture.",
        hint=FieldHint.architecture,
    )

    # Model dimensions
    # TODO: Remove (redundant default)
    expansion_factor: int = Field(
        default=2,
        desc="Expansion factor for Mamba blocks.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    # head_size [MambaLayer, Mamba2, DiscreteMamba2]
    state_size: int = Field(
        default=16,
        desc="State size for Mamba blocks.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    # [MambaLayer, Mamba2, DiscreteMamba2]
    conv_kernel_dimension: int = Field(
        default=4,
        desc="Conv kernel dimension for Mamba blocks.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    #  [MambaLayer, Mamba2]
    dt_rank: None | int = Field(
        default=None,
        desc="Rank of the Δ projection matrix. If 'None', will be set to ceil(hidden_size/16)",
        hint=FieldHint.architecture,
    )
    # head_groups [DiscreteMamba2]
    n_qk_heads: int = Field(
        default=32,
        desc="Number of QK heads for Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
    # heads [DiscreteMamba2]# TODO: Remove? (redundant)
    n_v_heads: int = Field(
        default=32,
        desc="Number of V heads for Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
    # c_size [MambaLayer, Mamba2, DiscreteMamba2]?
    d_inner: None | int = Field(
        default=None,
        desc="Inner dimension for Mamba2 blocks.",
        hint=FieldHint.core,
    )
    # xb_size [Mamba2]
    d_xb: int = Field(
        default=None,
        desc="Dimension of the xB in Mamba2 blocks.",
        hint=FieldHint.architecture,
    )

    # Model options
    # add_bias_linear [Mamba2, DiscreteMamba2] [hard-coded to False in MambaLayer]
    add_bias_linear: bool = Field(
        default=False,
        desc="Whether to use bias in SSM layers",
        hint=FieldHint.architecture,
    )
    # activation_type [DiscreteMamba2] [hard-coded to silu in MambaLayer, Mamba2]
    activation_type: ActivationType = Field(
        default=None,
        hint=FieldHint.architecture,
    )
    # repeat_xb_before_conv [Mamba2]
    repeat_kv_before_conv: bool = Field(
        default=True,
        desc="Whether to repeat x and B before (True) or after (False) the conv1d in Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
    # chunk_size [DiscreteMamba2]
    chunk_size: int = Field(
        default=256,
        desc="Chunk size for Mamba2 blocks.",
        hint=FieldHint.architecture,
    )

    # Learning rate
    # lr_scale [MambaLayer, Mamba2, DiscreteMamba2]
    mamba_lr_scale: float | None = Field(
        default=None,
        desc="Learning rate scale for Mamba blocks.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )

    # Initialization
    # dt_weight_initialization_method [Mamba2]
    dt_init: DTInitType = Field(
        default=DTInitType.random,
        desc="Initialization method for dt",
        hint=FieldHint.core,
    )
    # dt_weight_initialization_scale [Mamba2]
    dt_scale: float = Field(
        default=1.0,
        desc="Scale for dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    # dt_bias_initialization_min [MambaLayer, Mamba2]
    dt_min: float = Field(
        default=0.001,
        desc="Minimum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    # dt_bias_initialization_max [MambaLayer, Mamba2]
    dt_max: float = Field(
        default=0.1,
        desc="Maximum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    # dt_bias_initialization_floor [MambaLayer, Mamba2]
    dt_init_floor: float = Field(
        default=1e-4,
        desc="Minimum value for initializing dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.activation_type is None:
                self.activation_type = ActivationType.silu
        super()._validate()
        Assert.geq(self.dt_max, self.dt_min)

    def setup_tensor_space(self, tensor_space: TensorSpace, block_type: SSMBlockType) -> None:
        # Handled in the model.
        pass
