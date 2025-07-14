from fast_llm.models.ssm.external.apriel_15b_hybrid.modeling_ssm_hybrid_apriel15b import AprielSSMHybridForCausalLM

l3checkpoint = "/mnt/checkpoints/ssm/iterative_hybrids_only_new_layer_train/apriel_ssm_thinker15b_hybrid_3ssm_leastimportant_32h_init_rand"
model_l3 = AprielSSMHybridForCausalLM.from_pretrained(l3checkpoint, trust_remote_code=True, device_map="cpu")

mohawk_checkpoint = "/mnt/checkpoints_fml/final_stitched_model_L0-49"
model_mohawk = AprielSSMHybridForCausalLM.from_pretrained(mohawk_checkpoint, trust_remote_code=True, device_map="cpu")


layer_ids = "24"
sdm = model_mohawk.state_dict()
layer_sd = {k: v for k, v in sdm.items() if f"layers.{layer_ids}" in k}

r = model_l3.load_state_dict(layer_sd, strict=False)

print(r.unexpected_keys)

model_l3.save_pretrained(
    "/mnt/checkpoints/ssm/iterative_hybrids_only_new_layer_train/apriel_ssm_thinker15b_hybrid_3ssm_leastimportant_32h_init_mohawk_layer24"
)
