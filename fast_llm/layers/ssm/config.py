import enum
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.tensor_space import CompositeTensorDim, ConcatenatedTensorDim, TensorDim, TensorSpace
from fast_llm.engine.distributed.config import DistributedDimNames
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.config import LLMBlockConfig, NormalizationConfig
from fast_llm.utils import Assert, div

if typing.TYPE_CHECKING:
    from fast_llm.tensor import Initializer


class BaseSSMKwargs:
    _kwargs_attributes = {
        "cu_seqlens": "cu_seqlens",
        "seq_idx": "seq_idx",
        "ssm_position_ids": "ssm_position_ids",
    }

    _prefix = ""

    def __init_subclass__(cls, prefix="", **kwargs):
        super().__init_subclass__(**kwargs)
        cls._prefix = prefix
        for attr, value in BaseSSMKwargs._kwargs_attributes.items():
            setattr(cls, value, f"{cls._prefix}_{value}" if cls._prefix else value)


class SSMKwargs(BaseSSMKwargs, prefix=""):
    pass


class SSMDimNames:
    # TODO: Use separate tensor space for different mixers so there is no risk of name conflict.
    state = "ssm_state"  # State dimension (N), aka head size / num channels
    head_dim = "ssm_head_dim"
    head_groups = "ssm_head_groups"
    group_heads = "ssm_group_heads"
    conv1d_dim = "ssm_conv1d_dim"

    # Mamba 2
    x_proj_dim_2 = "x_proj_dim_2"  # d_xb
    convolution_kernel = "ssm_convolution_kernel"  # Kernel dimension of the conv1d in mamba layers

    dt_rank = "ssm_dt_rank"

    # Composite dimensions
    composite_heads = "ssm_composite_heads"
    composite_heads_and_head_dim = "ssm_composite_heads_and_head_dim"
    composite_heads_and_head_dim_nontp = "ssm_composite_heads_and_head_dim_nontp"
    composite_heads_and_state_dim = "ssm_composite_heads_and_state_dim"
    composite_head_groups_and_state = "ssm_composite_head_groups_and_state"
    composite_head_groups_and_head = "ssm_composite_head_groups_and_head"

    # Concatenated dimensions
    concatenated_convolution = "ssm_concatenated_convolution"
    concatenated_x_projection = "ssm_x_concatenated_x_projection"
    concatenated_inner_projection = "ssm_concatenated_inner_projection"


class SSMBlockType(enum.StrEnum):
    """
    An enum for the available mamba types for the MLP layer.
    """

    mamba = "m"
    mamba2_discrete = "m2d"
    mamba2 = "m2"
    transformer = "t"
    nemotron_h_mamba2 = "nm2"

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
        elif self == SSMBlockType.nemotron_h_mamba2:
            from fast_llm.layers.ssm.mamba2 import NemotronHMamba2

            return NemotronHMamba2
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

    # Nemotron H Mamba2 (the real mamba2 actually)
    # here instead of setting d_inner, we set head dim. and number of heads
    # Note: we do not implement n_groups for Mamba2, because, sicne we do MiL init, we do not want to share B and C parameters accross heads.
    # Instead, we mimic the GQA behaviour (x -> v, B -> k, C -> q), where x and B are shared accross heads. So this is the same as having n_groups = n_heads?
    n_groups: int = Field(
        default=128,
        desc="Number of groups for Mamba2. Allows sharing B and C parameters accross heads.",
        hint=FieldHint.architecture,
    )
    head_dim: int = Field(
        default=128,
        desc="Head dimension for Nemotron H",
        hint=FieldHint.architecture,
    )

    norm_before_gate: bool = Field(
        default=False,
        desc="Whether to apply the norm before the gate in Mamba2 blocks.",
        hint=FieldHint.architecture,
    )

    def _validate(self) -> None:
        with self._set_implicit_default():
            if self.activation_type is None:
                self.activation_type = ActivationType.silu
        super()._validate()
        Assert.geq(self.dt_max, self.dt_min)

    def setup_tensor_space(self, tensor_space: TensorSpace, block_type: SSMBlockType) -> None:
        tensor = tensor_space.distributed_config.get_distributed_dim(DistributedDimNames.tensor)

        # Head groups are configured differently depending on the block type.
        if block_type == SSMBlockType.mamba:
            num_heads = div(self.d_inner, self.state_size)
            num_head_groups = num_heads
        elif block_type == SSMBlockType.mamba2:
            num_heads = div(self.d_inner, self.state_size)
            num_head_groups = div(self.d_xb, self.state_size)
        elif block_type == SSMBlockType.nemotron_h_mamba2:
            # head dim and state size are not the same
            Assert.eq(self.head_dim, self.state_size)  # for MIL init
            num_heads = self.n_groups  # div(self.d_inner, self.head_dim)
            num_head_groups = div(self.d_xb, self.head_dim)
        elif block_type == SSMBlockType.mamba2_discrete:
            # TODO: Use different variables?
            num_heads = self.n_v_heads
            num_head_groups = self.n_qk_heads
        else:
            raise NotImplementedError(block_type)

        tensor_space.add_tensor_dim(state := TensorDim(SSMDimNames.state, self.state_size))
        if block_type == SSMBlockType.mamba2_discrete:
            tensor_space.add_tensor_dim(head_dim := TensorDim(SSMDimNames.head_dim, div(self.d_inner, num_heads)))
        elif block_type == SSMBlockType.nemotron_h_mamba2:
            tensor_space.add_tensor_dim(head_dim := TensorDim(SSMDimNames.head_dim, self.head_dim))
        else:
            head_dim = state

        tensor_space.add_tensor_dim(head_groups := TensorDim(SSMDimNames.head_groups, num_head_groups, tensor))
        tensor_space.add_tensor_dim(group_heads := TensorDim(SSMDimNames.group_heads, div(num_heads, num_head_groups)))
        tensor_space.add_tensor_dim(
            heads := CompositeTensorDim(SSMDimNames.composite_heads, (head_groups, group_heads))
        )
        # full d_inner or intermediate_size (e.g. for z gate, also the d_inner size for C in mamba2)
        tensor_space.add_tensor_dim(
            heads_and_head_dim := CompositeTensorDim(
                SSMDimNames.composite_heads_and_head_dim, (head_groups, group_heads, head_dim)
            )
        )
        # d_xb
        tensor_space.add_tensor_dim(
            head_groups_and_head := CompositeTensorDim(
                SSMDimNames.composite_head_groups_and_head, (head_groups, head_dim)
            )
        )
        tensor_space.add_tensor_dim(TensorDim(SSMDimNames.convolution_kernel, self.conv_kernel_dimension))

        # DT projection
        if block_type in (SSMBlockType.mamba, SSMBlockType.mamba2):
            tensor_space.add_tensor_dim(dt_rank := TensorDim(SSMDimNames.dt_rank, self.dt_rank))

        if block_type == SSMBlockType.mamba:
            tensor_space.add_tensor_dim(
                ConcatenatedTensorDim(SSMDimNames.concatenated_x_projection, (dt_rank, state, state))
            )
            # TODO: Use composition instead
            tensor_space.add_tensor_dim(
                ConcatenatedTensorDim(
                    SSMDimNames.concatenated_inner_projection, (heads_and_head_dim, heads_and_head_dim)
                )
            )
        elif block_type == SSMBlockType.mamba2:
            # TODO: Factor out state?
            tensor_space.add_tensor_dim(
                ConcatenatedTensorDim(
                    SSMDimNames.concatenated_inner_projection,
                    (heads_and_head_dim, head_groups_and_head, head_groups_and_head, heads_and_head_dim),
                )
            )
        elif block_type == SSMBlockType.nemotron_h_mamba2:
            # for the norm
            tensor_space.add_tensor_dim(
                TensorDim(
                    SSMDimNames.composite_heads_and_head_dim_nontp, num_head_groups * group_heads.size * head_dim.size
                )
            )
            # state and head dim are not the same
            # C: for each head, size of state
            tensor_space.add_tensor_dim(
                heads_and_state_dim := CompositeTensorDim(
                    SSMDimNames.composite_heads_and_state_dim, (head_groups, group_heads, state)
                )
            )
            # B: for each head group, size of state
            tensor_space.add_tensor_dim(
                head_groups_and_state := CompositeTensorDim(
                    SSMDimNames.composite_head_groups_and_state, (head_groups, state)
                )
            )
            # here we apply depthwise conv. layer to xBC, so the dim. is x (d_xb) x B (d_bb) x C
            tensor_space.add_tensor_dim(
                conv1d_dim := ConcatenatedTensorDim(
                    SSMDimNames.conv1d_dim, (heads_and_state_dim, head_groups_and_head, head_groups_and_state)
                )
            )

            # inner projection dimention: also includes z (gate), which has size d_inner (heads_and_head_dim)
            tensor_space.add_tensor_dim(
                ConcatenatedTensorDim(
                    SSMDimNames.concatenated_inner_projection,
                    (conv1d_dim, heads_and_head_dim),
                )
            )
        elif block_type == SSMBlockType.mamba2_discrete:
            tensor_space.add_tensor_dim(
                ConcatenatedTensorDim(
                    SSMDimNames.concatenated_inner_projection,
                    (heads_and_head_dim, head_groups_and_state, head_groups_and_state, heads_and_head_dim, heads),
                )
            )
            tensor_space.add_tensor_dim(
                ConcatenatedTensorDim(
                    SSMDimNames.concatenated_convolution,
                    (heads_and_head_dim, head_groups_and_state, head_groups_and_state),
                )
            )
