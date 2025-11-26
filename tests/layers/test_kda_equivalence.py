import torch

from fast_llm.engine.distributed.distributed import Distributed

try:
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextConfig, Qwen3NextGatedDeltaNet
except ImportError:
    Qwen3NextConfig, Qwen3NextGatedDeltaNet = None, None


def _materialize_mixer_tensors(module: torch.nn.Module, distributed: Distributed, device: torch.device) -> None:
    """
    Instantiate meta-allocated parameters on the requested device so the layer can run standalone.
    """
    for name, param in module.named_parameters():
        if param.device.type != "meta":
            continue
        param_data = torch.empty_like(param, device=device)
        param.init_parameter(param_data, distributed)
        module_path, param_name = name.rsplit(".", 1) if "." in name else (None, name)
        target = module
        if module_path is not None:
            for part in module_path.split("."):
                target = getattr(target, part)
        new_param = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
        new_param.grad = None
        new_param.grad_buffer = torch.zeros_like(param_data)
        new_param.param_grad_is_zero = True
        target._parameters[param_name] = new_param
