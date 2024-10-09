BASE_JOB_NAME="mistral-7b-FastLLM-stardoc-debug-local"

export PYTHONHASHSEED=12345

export MODEL_ARGS="\
--pretrained_checkpoint_type=huggingface \
--pretrained_checkpoint_path=/data/git/Fast-LLM/stardoc_hf/stardoc_checkpoint \
--use_pretrained_config=1 \
--attention_dropout=0.0 \
--hidden_dropout=0.0 \
--max_num_images=5 \
--image_resolution=224 \
--num_image_tokens=256 \
--image_encoder_hidden_size=1024 \
--image_encoder_type=clip \
"

export STAGE_ARGS="\
--zero_stage=3 \
"

export OPTIMIZER_ARGS="\
--lr=0.000001 \
--lr_decay_style=cosine \
--lr_decay_iters=250 \
--lr_warmup_iters=100 \
--min_lr=0.0 \
--weight_decay=0.1 \
--adam_beta1=.9 \
--adam_beta2=.95 \
--clip_grad=1.0 \
"

export DATA_ARGS="\
--split=9998,2,0 \
--dataset_source=multimodal \
--data_path=/data/datasets/stardoc/BigDoc-MultiTurn-v0.3 \
--tokenizer_type=PreTrainedTokenizer \
--tokenizer_path=/data/models/mistral/HF/Mistral-7B-v0.3 \
"

export SCHEDULE_ARGS="\
--batch_size=32 \
--micro_batch_size=1 \
--sequence_length=8192 \
"

export DISTRIBUTED_ARGS="\
--training_dtype=bf16 \
--distributed_timeout=600 \
--seed=984059 \
--sequence_data_parallel=1 \
"

export RUN_ARGS="\
--log_interval=10 \
--log_offset=0 \
--checkpoint_interval=500 \
--max_checkpoints=5 \
--export_interval=25000 \
"

export TRAINING_ARGS="\
--train_iters=40000 \
--validation_iters=25000000 \
--validation_interval=1000000 \
--test_iters=0 \
--num_workers=1 \
"

export ALL_ARGS="\
$MODEL_ARGS \
$STAGE_ARGS \
$DATA_ARGS \
$SCHEDULE_ARGS \
$OPTIMIZER_ARGS \
$DISTRIBUTED_ARGS \
$TRAINING_ARGS \
$RUN_ARGS \
"

torchrun --nproc-per-node=8 \
    --log-dir=output/$BASE_JOB_NAME/logs \
    --redirects=3 \
    pretrain_fast_llm.py $ALL_ARGS --experiment_dir="output/$BASE_JOB_NAME/"

# torchrun --nproc-per-node=8 \
#     pretrain_fast_llm.py $ALL_ARGS --experiment_dir="output/$BASE_JOB_NAME/"