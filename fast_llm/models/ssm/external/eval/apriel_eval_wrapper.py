from typing import Optional, Union

import lm_eval.models.utils
import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("apriel_ssm")
class AprielSSMWrapper(HFLM):
    """Wrapper for Rene model for compatibility with lm-evaluation-harness."""

    def __init__(self, pretrained, **kwargs) -> None:
        if "backend" in kwargs:
            # rene currently only supports causal models
            assert kwargs["backend"] == "causal"

        super().__init__(
            pretrained=pretrained,
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", "/mnt/checkpoints/upstream/Mistral-Nemo-Base-2407/"),
            max_length=kwargs.pop("max_length", 4096),
            **kwargs,
        )

    def _get_config(self, pretrained: str, **kwargs) -> None:
        """Get the model configuration."""
        from fast_llm.models.ssm.external.aperiel_ssm.configuration_ssm_apriel import AprielSSMConfig

        self._config = AprielSSMConfig.from_pretrained(pretrained)

    def _create_model(self, pretrained: str, dtype: Optional[Union[str, torch.dtype]] = "float16", **kwargs) -> None:
        """Create the model."""
        from fast_llm.models.ssm.external.aperiel_ssm.modeling_ssm_apriel import AprielSSMForCausalLM

        self._model = AprielSSMForCausalLM.from_pretrained(
            pretrained,
            device=self._device,
            dtype=torch.bfloat16 if dtype == "auto" else lm_eval.models.utils.get_dtype(dtype),
            trust_remote_code=True,
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Generate text from the model."""
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
    """Wrapper for Rene model for compatibility with lm-evaluation-harness."""

    def __init__(self, pretrained, **kwargs) -> None:
        if "backend" in kwargs:
            # rene currently only supports causal models
            assert kwargs["backend"] == "causal"

        super().__init__(
            pretrained=pretrained,
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", "/mnt/checkpoints/upstream/Mistral-Nemo-Base-2407/"),
            max_length=kwargs.pop("max_length", 4096),
            **kwargs,
        )

    def _get_config(self, pretrained: str, **kwargs) -> None:
        """Get the model configuration."""
        from fast_llm.models.ssm.external.apriel_hybrid.configuration_ssm_hybrid_apriel import AprielSSMHybridConfig

        self._config = AprielSSMHybridConfig.from_pretrained(pretrained)

    def _create_model(self, pretrained: str, dtype: Optional[Union[str, torch.dtype]] = "float16", **kwargs) -> None:
        """Create the model."""
        from fast_llm.models.ssm.external.apriel_hybrid.modeling_ssm_hybrid_apriel import AprielSSMHybridForCausalLM

        self._model = AprielSSMHybridForCausalLM.from_pretrained(
            pretrained,
            device=self._device,
            dtype=torch.bfloat16 if dtype == "auto" else lm_eval.models.utils.get_dtype(dtype),
            trust_remote_code=True,
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """Generate text from the model."""
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
