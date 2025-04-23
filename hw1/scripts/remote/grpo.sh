export HOST_IP=$MASTER_ADDR
export NCCL_CUMEM_ENABLE=0
export TOKENIZERS_PARALLELISM=False

pkill -f ray

cleanup() {
    echo "Cleaning up..."
    pkill -f "serve_rm"
    ray stop
}
trap cleanup EXIT

ROLLOUT_BS=64
N_SAMPLES_PER_PROMPT=8
TEMPERATURE=0.85
NUM_EPISODES=10
KL_COEF=0.01
BS=256
EP=1
LR=1e-6
EVAL_STEPS=10
MAX_GEN_LEN=1024

TRIAL_NAME=grpo_4 # 选用long_sft_1_orig，看来过多sft并非好事
# grpo_4是最终决战，训一个0.85，使用0.25less，训一个0.85但0.3less的，复刻一下前面成功的就结束了

# wandb setting
export WANDB_MODE=online
export WANDB_DIR=/mnt/data/jianghantao/Minimum-Implementation-of-O1/hw1/.wandb_logs_grpo

DATA_PATH=/mnt/data/jianghantao/Minimum-Implementation-of-O1/hw1/data/train/math3k_rl_prompt
# model path
POLICY_MODEL_PATH=/mnt/data/jianghantao/Minimum-Implementation-of-O1/ckpts/long_sft_1_orig

SAVE_PATH=/mnt/data/jianghantao/Minimum-Implementation-of-O1/ckpts/${TRIAL_NAME}
SAMPLES_SAVE_PATH=/mnt/data/jianghantao/Minimum-Implementation-of-O1/hw1/data/output/rl/${TRIAL_NAME}

# start rm
python -m src.cli.serve_rm \
    --mode rule \
    --tokenizer_path $POLICY_MODEL_PATH \
    --max_gen_len $MAX_GEN_LEN \
    --data_path $DATA_PATH \
    --port 5000 &

sleep 10s
RAY_MASTER_PORT=6379
RAY_DASHBOARD_PORT=8265
								
ray start --head --port=$RAY_MASTER_PORT --dashboard-host=127.0.0.1 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 4

							 
sleep 10s
# replace working_dir with your own working dir
RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit --address="http://127.0.0.1:12345" \
    --runtime-env-json='{"working_dir": "/mnt/data/jianghantao/Minimum-Implementation-of-O1/hw1", "conda": "hw1", "excludes": ["data/output/rl/grpo_test*/","data/output/rl/grpo_temp_*","data/output/rl/grpo_0/", "data/output/rl/grpo_1/", "data/output/rl/grpo_temp_9", "data/train/math3k_cot.jsonl", "data/train/math3k_longcot.jsonl", ".wandb_logs_sft_*/"]}' \
    -- python3 -m src.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --eval_steps $EVAL_STEPS \
    --save_steps 10 \
    --pretrain $POLICY_MODEL_PATH \
    --ref_pretrain $POLICY_MODEL_PATH \
    --remote_rm_url http://localhost:5000/get_reward \
    --save_path $SAVE_PATH \
    --ckpt_path $SAVE_PATH \
    --samples_save_path $SAMPLES_SAVE_PATH \
    --micro_train_batch_size 8 \
    --train_batch_size $BS \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size $ROLLOUT_BS \
    --n_samples_per_prompt $N_SAMPLES_PER_PROMPT \
    --max_epochs $EP \
    --num_episodes $NUM_EPISODES \
    --prompt_max_len 350 \
    --generate_max_len $MAX_GEN_LEN \
    --advantage_estimator grpo \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate $LR \
    --lr_warmup_steps 10 \
    --init_kl_coef $KL_COEF \
    --prompt_data $DATA_PATH \
    --test_path /mnt/data/jianghantao/Minimum-Implementation-of-O1/hw1/data/eval/RL.jsonl \
    --input_key context_messages \
    --apply_chat_template \
    --max_samples 100000 \
    --packing_samples \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --vllm_sync_backend nccl \
    --gradient_checkpointing \
    --temperature $TEMPERATURE \
    --use_wandb 3eb7f9637fc1a5a7903f3c4acb034dce0d346dd4 \
    --wandb_project sjtu_cs2916_grpo \
    --wandb_group rl.grpo \
    --wandb_run_name $TRIAL_NAME &
wait   
