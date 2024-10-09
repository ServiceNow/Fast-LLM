# Required or optional environment variables
export PROJECT_DIR="/mnt/akshay/stardoc-FastLLM/Fast-LLM/output"
export PROJECT_NAME="stardoc-debug"
export PROJECT_VERSION="1.0"
# export DATA_PATH_LIST=
export DATA_PATH="/mnt/stardoc/datasets/save_hf/BigDoc-MultiTurn-v0.3"
# export DATA_PATH_JSON=
# export PRETRAINED_MISTRAL_PATH=
# export PRETRAINED_MIXTRAL_PATH=
export PRETRAINED_STARDOC_PATH="/mnt/akshay/stardoc-FastLLM/Fast-LLM/stardoc_hf_model/stardoc_checkpoint"
export TOKENIZER_PATH="/mnt/core_llm/models/mistral/HF/Mistral-7B-v0.3"

export PYTHONHASHSEED=12345

export CMD_ARGS="fast-llm train stardoc"
# export CMD_ARGS="python /mnt/akshay/stardoc-FastLLM/Fast-LLM/fast_llm/tools/train.py stardoc"

export MODEL_ARGS_PRETRAINED="\
--pretrained_checkpoint_type=huggingface \
--pretrained_checkpoint_path=$PRETRAINED_STARDOC_PATH \
--use_pretrained_config=1 \
"

export MODEL_ARGS_ARCHITECTURE="\
--num_layers=32 \
--hidden_size=4096 \
--vocab_size=32000 \
--num_attention_heads=32 \
--head_groups=8 \
--add_linear_biases=0 \
--ffn_hidden_size=14336 \
--kv_channels=128 \
--use_rotary_embeddings=1 \
--rotary_embedding_scale=-9.210340371976184 \
--gated=1 \
--activation_type=silu \
--normalization_type=rms_norm \
--tie_word_embeddings=0 \
--window_size=8192 \
"

export MULTIMODAL_ARGS ="\
--image_encoder_hidden_size=1024 \
--num_image_tokens=256 \
--max_num_images=7 \
--image_encoder_type=clip \
"

export DATA_ARGS="\
--split=9998,2,0 \
--dataset_type=stardoc \
--dataset_source=multimodal \
--data_path=$DATA_PATH \
--tokenizer_type=PreTrainedTokenizer \
--tokenizer_path=$TOKENIZER_PATH \
"

export TRAINING_ARGS="\
--batch_size=32 \
--sequence_length=8192 \
--train_iters=500000 \
--weight_decay=0.1 \
--adam_beta1=0.9 \
--adam_beta2=0.95 \
--clip_grad=1.0 \
--lr=0.0001 \
--lr_warmup_iters=1000 \
--lr_decay_style=cosine \
--lr_decay_iters=500000 \
--min_lr=0.000003 \
"

export PERFORMANCE_ARGS="\
--micro_batch_size=4 \
--training_dtype=bf16 \
--zero_stage=2 \
--num_workers=8 \
"

export MONITORING_ARGS="\
--validation_iters=25 \
--validation_interval=1000 \
--log_interval=10 \
--log_offset=0 \
--checkpoint_interval=500 \
--max_checkpoints=5 \
--export_interval=25000 \
"
# --wandb_status_interval=25000 \
# --wandb_entity_name=$WANDB_ENTITY_NAME \
# --wandb_project_name=$PROJECT_NAME \
# --wandb_group_name=$PROJECT_VERSION \
# "

export ALL_ARGS="\
$CMD_ARGS \
$MODEL_ARGS_PRETRAINED \
$MULTIMODAL_ARGS \
$DATA_ARGS \
$TRAINING_ARGS \
$PERFORMANCE_ARGS \
$MONITORING_ARGS \
"

export PROFILE_ARGS="\
--profile_cuda=1 \
--profile_skip=10 \
--profile_wait=95 \
--profile_warmup=2 \
--profile_cycles=3 \
--profile_export=1 \
"


run_local () { # run(name, num_gpus, base_cmd)
  echo $1 $2 $3
  export TORCHRUN="torchrun --nproc-per-node=$2 --nnodes=1 --no-python"
  $TORCHRUN $3 --experiment_dir=$PROJECT_DIR/$PROJECT_NAME_$PROJECT_VERSION/$1
}

run_c10d () { # run(name, num_nodes, base_cmd)
  echo $1 $2 $3
  export TORCHRUN="torchrun --nproc-per-node=8 --nnodes=$2 --no-python --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR"
  $TORCHRUN $3 --experiment_dir=$PROJECT_DIR/$PROJECT_NAME_$PROJECT_VERSION/$1
}

run_local debug 8 "$ALL_ARGS"
# run_c10d mistral_example 16 "$ALL_ARGS"
# run_c10d mixtral_example 16 "$ALL_ARGS $MIXTRAL_ARGS --train_iters=50"
