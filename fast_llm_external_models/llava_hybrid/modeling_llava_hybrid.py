from torch import nn
from transformers import AutoModel, LlavaForConditionalGeneration, LlavaModel
from transformers.activations import ACT2FN

from .configuration_llava_hybrid import LlavaHybridConfig

try:
    # In the fast-llm repo, import from the SSM modeling file
    from fast_llm_external_models.apriel_hybrid_ssm.modeling_apriel_hybrid_ssm import (
        AprielThinkerSSMHybridModel,
        HybridMambaAttentionDynamicCache,
    )
except ImportError:
    # In the exported checkpoint, import from local file
    from .modeling_apriel_hybrid_ssm import AprielThinkerSSMHybridModel, HybridMambaAttentionDynamicCache


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaHybridConfig):
        super().__init__()
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.projector_intermediate_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_intermediate_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaHybridModel(LlavaModel):
    """
    Llava SSM-Hybrid-decoder model.
    """

    config_class = LlavaHybridConfig

    def __init__(self, config: LlavaHybridConfig):
        super(LlavaModel, self).__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        assert (
            config.text_config.model_type == "apriel_ssm_thinker_hybrid"
        ), "Only Apriel SSM Hybrid model is supported in LlavaHybridModel"

        self.language_model = AprielThinkerSSMHybridModel(config.text_config)
        self.post_init()


class LlavaHybridForConditionalGeneration(LlavaForConditionalGeneration):
    config_class = LlavaHybridConfig

    def __init__(self, config: LlavaHybridConfig):
        super(LlavaForConditionalGeneration, self).__init__(config)
        self.model = LlavaHybridModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        **kwargs,
    ):
        # Copy of the method from `AprielThinkerSSMHybridForCausalLM`
        # Overwritten -- has a unique cache type, `HybridMambaAttentionDynamicCache`

        empty_past_kv = past_key_values is None or not isinstance(past_key_values, HybridMambaAttentionDynamicCache)

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if not empty_past_kv:
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config.text_config, input_ids.shape[0], self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        # Copy from `LlavaForConditionalGeneration.prepare_inputs_for_generation`
        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                # "logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": cache_position,
            }
        )
        return model_inputs
