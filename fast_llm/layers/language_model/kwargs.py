from fast_llm.layers.block.config import BlockKwargs


class TargetsKwargs:
    lm_target = "preprocessed_lm_target"
    dpo_target = "preprocessed_dpo_target"
    reference_model_logits = "reference_model_logits"
    dpo_reference_model_logits = "dpo_reference_model_logits"


class LanguageModelKwargs(BlockKwargs):
    token_ids = "token_ids"
    position_ids = "position_ids"
    token_map = "token_map"
    sample_map = "sample_map"
    embedding_map = "embedding_map"
    # TODO: These are generic
    labels = "labels"
    phase = "phase"
    chosen_spans = "chosen_spans"
    rejected_spans = "rejected_spans"
    loss_mask = "loss_mask"
    mask_inputs = "mask_inputs"
