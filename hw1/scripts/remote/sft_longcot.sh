set -x

export NCCL_CUMEM_ENABLE=0
export WANDB_MODE=online
export WANDB_DIR=./.wandb_logs_sft_longcot
export WANDB_KEY=3eb7f9637fc1a5a7903f3c4acb034dce0d346dd4

# current: long
# next plans: first orig, then long, dont know how to mix
# 最好的做法是，先短cot开始训练3个周期，再用更长的max_len训练3个周期
# 目前longcot_orig是EP=5, LR=5e-6, max_len=16384的
# 即将训练：longcot_long，基于longcot_orig的ckpt，max_len=20480, LR=4e-6, EP=2
# 这不会过拟合吧？

# BS=256
# EP=2
# LR=5e-6 # 1e-5

# TRIAL_NAME=sft_longcot_longlr
# MODEL_PATH=../ckpts/longcot_sft_biglr
# SAVE_PATH=../ckpts/longcot_sft_long
# DATA_PATH=./data/train/math3k_longcot.jsonl

# 策略：以EP=4的sft模型为基础，其他不变，训EP=6的长模型

BS=256 # 256
EP=5 # 5
LR=1e-5 # 1e-5

TRIAL_NAME=sft_longcot_1_orig
MODEL_PATH=../ckpts/sft_1
SAVE_PATH=../ckpts/long_sft_1_orig
DATA_PATH=./data/train/math3k_longcot.jsonl

# revised max_len 16384
# ckpts/longcot_sft_long: use max_len=24576,  LR=5e-6, EP=4，配合截断短数据集使用
# ckpts/longcot_sft_orig: use max_len=16384, LR=5e-6, EP=5
# 难道学习率真的要从1e-5开始？可是训练5期是不是太多了？
# 还是说，提高max_len?长问题的比例其实还蛮大

read -r -d '' training_commands <<EOF
src.cli.train_sft \
   --max_len 20480 \
   --dataset $DATA_PATH \
   --input_key prompt \
   --output_key solution \
   --train_batch_size $BS \
   --micro_train_batch_size 1 \
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
