import math
from typing import Optional

from fast_llm.config import Field, FieldHint, FieldUpdate, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.layers.common.config import NormalizationConfig
from fast_llm.layers.transformer.config import TransformerArchitectureConfig
from fast_llm.utils import Assert

@config_class()
class MambaConfig(TransformerArchitectureConfig, BaseModelConfig):
    """Configuration for a Structured State Space Model (SSM) layer."""
    
    # Core architecture parameters
    hidden_size: int = Field(
        default=768,
        desc="Size of the hidden representations",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    
    state_size: int = Field(
        default=16,
        desc="Size of the internal state vector",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    
    expansion_factor: int = Field(
        default=2,
        desc="Factor by which to expand hidden size in SSM computation",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    # SSM specific parameters
    conv_dimension: int = Field(
        default=4,
        desc="Size of the convolutional kernel",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    dt_rank: str | int = Field(
        default="auto",
        desc="Rank of the Δ projection matrix. If 'auto', set to ceil(hidden_size/16)",
        hint=FieldHint.core,
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

    # Layer parameters
    add_bias_linear: bool = Field(
        default=False,
        desc="Whether to use bias in linear transformations",
        hint=FieldHint.core,
    )

    conv_bias: bool = Field(
        default=True,
        desc="Whether to use bias in convolution layer",
        hint=FieldHint.core,
    )

    # Normalization
    normalization: NormalizationConfig = FieldUpdate(
        default_factory=NormalizationConfig
    )

    # Performance optimization
    use_fast_path: bool = Field(
        default=True,
        desc="Whether to use optimized CUDA kernels when available",
        hint=FieldHint.performance,
    )

    # Initialization parameters
    init_method_std: float = Field(
        default=None,
        desc="Default scale for weight initialization. Default: hidden_size**-0.5",
        hint=FieldHint.optional,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.init_method_std is None:
            self.init_method_std = self.hidden_size**-0.5

        super()._validate()
        
        # Validate SSM-specific parameters
        Assert.gt(self.state_size, 0)
        Assert.gt(self.expansion_factor, 0)
        Assert.gt(self.conv_dimension, 0)
        Assert.gt(self.dt_min, 0)
        Assert.gt(self.dt_max, 0)
        Assert.gt(self.dt_init_floor, 0)
        Assert.geq(self.dt_max, self.dt_min)
        
        if isinstance(self.dt_rank, int):
            Assert.gt(self.dt_rank, 0)