import os
import pathlib

import pytest
import torch

from fast_llm.layers.ssm.config import SSMConfig
from fast_llm.layers.transformer.config import TransformerConfig
from fast_llm.models.ssm.config import HybridSSMBaseModelConfig

TEST_RESULTS_PATH = pathlib.Path(os.environ.get("TEST_RESULTS_PATH", "/tmp/fast_llm_tests")).resolve()
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


def materialize_meta_tensors(model, tensor_space):
    # Materialize parameters that are on meta device
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            # Check if the parameter is a custom tensor type
            if hasattr(param, "tensor_name") and hasattr(param, "init_parameter"):
                param_data = param.new_empty(param.shape, device="cuda")
                # Initialize param_data
                param.init_parameter(param_data, tensor_space.distributed)
                # Replace the parameter in the module
                module_path, param_name = name.rsplit(".", 1) if "." in name else (None, name)
                module = model
                if module_path is not None:
                    for part in module_path.split("."):
                        module = getattr(module, part)
                param = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
                # TODO: add param_grad_is_zero etc., grad_buffer, etc., see test_mlp_recomputation
                param.grad = None
                param.grad_buffer = torch.empty_like(param)
                param.param_grad_is_zero = True
                module._parameters[param_name] = param
    return model


def get_hybrid_config(hybrid_block_layout=["t", "m"], prediction_heads=1, default_mtp_type=None):
    config = HybridSSMBaseModelConfig(
        transformer=TransformerConfig(num_layers=len(hybrid_block_layout)),
        ssm=SSMConfig(),
        hybrid_block_layout=hybrid_block_layout,
        prediction_heads=prediction_heads,
        default_mtp_type=default_mtp_type,
        init_method_std_embed=0.02,
        init_method_min_embed=-0.02,
        init_method_max_embed=0.02,
        use_position_embeddings=True,
        tie_word_embeddings=False,
    )
    return config
