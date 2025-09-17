from typing import Optional, Union

import lm_eval.models.utils
import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


def _get_device():
    """Get the correct device for distributed training."""
    # Check if we're in a distributed setting with accelerate
    if hasattr(torch.distributed, "is_initialized") and torch.distributed.is_initialized():
        # Use the local rank as device
        local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        # Use current device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")

    return device


def _move_tensors_to_device(obj, device):
    """Recursively move tensors in nested structures to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_tensors_to_device(item, device) for item in obj)
    else:
        return obj


@register_model("apriel_ssm")
class AprielSSMWrapper(HFLM):
    """Wrapper for AprielSSM model for compatibility with lm-evaluation-harness."""

    def __init__(self, pretrained, **kwargs) -> None:
        if "backend" in kwargs:
            assert kwargs["backend"] == "causal"

        super().__init__(
            pretrained=pretrained,
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", "/mnt/checkpoints/upstream/Apriel-5B-Instruct/"),
            **kwargs,
        )

        # Override device detection for distributed settings
        self._device = _get_device()

    def _get_config(self, pretrained: str, **kwargs) -> None:
        """Get the model configuration."""
        from fast_llm_external_models.apriel_ssm.configuration_ssm_apriel import AprielSSMConfig

        self._config = AprielSSMConfig.from_pretrained(pretrained)

    def _create_model(self, pretrained: str, dtype: Optional[Union[str, torch.dtype]] = "float16", **kwargs) -> None:
        """Create the model."""
        from fast_llm_external_models.apriel_ssm.modeling_ssm_apriel import AprielSSMForCausalLM

        # Ensure we're using the correct device
        device = _get_device()
        self._device = device

        self._model = AprielSSMForCausalLM.from_pretrained(
            pretrained,
            device=device,
            dtype=torch.bfloat16 if dtype == "auto" else lm_eval.models.utils.get_dtype(dtype),
            trust_remote_code=True,
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Generate text from the model."""
        # Ensure we're using the correct device
        device = _get_device()

        # Ensure context is on the same device as the model
        context = context.to(device)

        # Move any tensors in generation_kwargs to the correct device
        generation_kwargs = _move_tensors_to_device(generation_kwargs, device)

        for key in ("do_sample", "attention_mask"):
            if key in generation_kwargs:
                generation_kwargs.pop(key)

        # The custom GenerationMixin imported from mamba_ssm currently does not support
        # passing stopping criteria.
        # For the time being, we simply generate to max length, then truncate (equivalent result).
        # This should be revisited to speed up generation
        # stopping_criteria = stop_sequences_criteria(self.tokenizer, stop, 1, context.shape[0])

        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            **generation_kwargs,
        )


@register_model("apriel_hybrid_ssm")
class AprielHybridSSMWrapper(HFLM):
    """Wrapper for AprielHybridSSM model for compatibility with lm-evaluation-harness."""

    def __init__(self, pretrained, **kwargs) -> None:
        if "backend" in kwargs:
            assert kwargs["backend"] == "causal"

        super().__init__(
            pretrained=pretrained,
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", "/mnt/checkpoints/upstream/Apriel-5B-Instruct/"),
            **kwargs,
        )

        # Override device detection for distributed settings
        self._device = _get_device()

    def _get_config(self, pretrained: str, **kwargs) -> None:
        """Get the model configuration."""
        from fast_llm_external_models.apriel_hybrid.configuration_ssm_hybrid_apriel import AprielSSMHybridConfig

        self._config = AprielSSMHybridConfig.from_pretrained(pretrained, trust_remote_code=True)

    def _create_model(self, pretrained: str, dtype: Optional[Union[str, torch.dtype]] = "float16", **kwargs) -> None:
        """Create the model."""
        from fast_llm_external_models.apriel_hybrid.modeling_ssm_hybrid_apriel import AprielSSMHybridForCausalLM

        # Ensure we're using the correct device
        device = _get_device()
        self._device = device

        self._model = AprielSSMHybridForCausalLM.from_pretrained(
            pretrained,
            device=device,
            torch_dtype=torch.bfloat16 if dtype == "auto" else lm_eval.models.utils.get_dtype(dtype),
            **kwargs,
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # Ensure we're using the correct device
        device = _get_device()

        # Ensure context is on the same device as the model
        context = context.to(device)

        # Move any tensors in generation_kwargs to the correct device
        generation_kwargs = _move_tensors_to_device(generation_kwargs, device)

        stopping_criteria = lm_eval.models.utils.stop_sequences_criteria(
            self.tokenizer,
            stop,
            context.shape[1],
            context.shape[0],
        )

        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            use_cache=True,
            **generation_kwargs,
        )


@register_model("apriel_hybrid_ssm_15b")
class AprielHybrid15bSSMWrapper(HFLM):
    """Wrapper for AprielHybridSSM model for compatibility with lm-evaluation-harness."""

    def __init__(self, pretrained, **kwargs) -> None:
        if "backend" in kwargs:
            assert kwargs["backend"] == "causal"

        super().__init__(
            pretrained=pretrained,
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", "/mnt/checkpoints/upstream/Apriel-Nemotron-15b-Thinker"),
            **kwargs,
        )

        # Override device detection for distributed settings
        self._device = _get_device()

    def _get_config(self, pretrained: str, **kwargs) -> None:
        """Get the model configuration."""
        from fast_llm_external_models.apriel_15b_hybrid.configuration_ssm_hybrid_apriel15b import AprielSSMHybridConfig

        self._config = AprielSSMHybridConfig.from_pretrained(pretrained, trust_remote_code=True)

    def _create_model(self, pretrained: str, dtype: Optional[Union[str, torch.dtype]] = "float16", **kwargs) -> None:
        """Create the model."""
        from fast_llm_external_models.apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import (
            AprielThinkerSSMHybridForCausalLM,
        )

        # Ensure we're using the correct device
        device = _get_device()
        self._device = device

        self._model = AprielThinkerSSMHybridForCausalLM.from_pretrained(
            pretrained,
            device=device,
            torch_dtype=torch.bfloat16 if dtype == "auto" else lm_eval.models.utils.get_dtype(dtype),
            **kwargs,
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # Ensure we're using the correct device
        device = _get_device()

        # Ensure context is on the same device as the model
        context = context.to(device)
        self.model.to(device)

        # Move any tensors in generation_kwargs to the correct device
        generation_kwargs = _move_tensors_to_device(generation_kwargs, device)

        stopping_criteria = lm_eval.models.utils.stop_sequences_criteria(
            self.tokenizer,
            stop,
            context.shape[1],
            context.shape[0],
        )

        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            use_cache=True,
            **generation_kwargs,
        )
