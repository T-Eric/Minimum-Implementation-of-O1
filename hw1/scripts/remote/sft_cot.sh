set -x

export NCCL_CUMEM_ENABLE=0
export WANDB_MODE=online
export WANDB_DIR=./.wandb_logs_sft_cot
export WANDB_KEY=3eb7f9637fc1a5a7903f3c4acb034dce0d346dd4

BS=256 # 256
EP=5 # 3, 据说提高EP就能训出更好的sft_cot
LR=1e-5 # 1e-5

TRIAL_NAME=sft_cot_long_epoch_1
MODEL_PATH=/mnt/data/Qwen2.5-Math-1.5B
SAVE_PATH=../ckpts/sft_1
DATA_PATH=./data/train/math3k_cot.jsonl

read -r -d '' training_commands <<EOF
src.cli.train_sft \
   --max_len 2048 \
   --dataset $DATA_PATH \
   --input_key prompt \
   --output_key solution \
   --train_batch_size $BS \
   --micro_train_batch_size 4 \
   --apply_chat_template \
   --max_samples 50000000 \
   --pretrain $MODEL_PATH \
   --save_path $SAVE_PATH \
   --ckpt_path $SAVE_PATH \
   --disable_ds_ckpt \
   --max_ckpt_num 100 \
   --save_hf_ckpt \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs $EP \
   --bf16 \
   --flash_attn \
   --learning_rate $LR \
   --lr_scheduler cosine_with_min_lr \
   --gradient_checkpointing \
   --packing_samples \
   --use_wandb $WANDB_KEY \
   --wandb_project sjtu_cs2916_baseline \
   --wandb_group sft \
   --wandb_run_name $TRIAL_NAME 
EOF

torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 \
    --master_addr "127.0.0.1" --master_port 12345 -m ${training_commands}
