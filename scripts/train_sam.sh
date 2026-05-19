#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/train_grpo_rollout2.out
#SBATCH -e log/error-train_grpo_rollout2.out
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --job-name=train_grpo_rollout2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -G A100:4

export VLLM_DISABLE_COMPILE_CACHE=1
source ~/.bashrc
conda activate prm_rlvr
datasets=(gsm8k)
types=(constant-1e3)
lrs=(1e-3)
rollout_ns=(16)

for dataset in "${datasets[@]}";do
for type in "${types[@]}";do
for rollout_n in "${rollout_ns[@]}";do
for lr in "${lrs[@]}";do
model_name=OLMo-150M-${type}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus_baseline \
    algorithm.norm_adv_by_std_in_grpo=false\
    data.train_files=data/${dataset}/train.parquet \
    data.val_files=['data/math/test.parquet']\
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1536\
    data.filter_overlong_prompts=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    +actor_rollout_ref.actor.use_sam=true \
    data.truncation=right \
    actor_rollout_ref.model.path=OLMo-150M/${model_name} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.optim_name='SAM' \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=1e-4 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=false \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${rollout_n}\
    actor_rollout_ref.rollout.temperature=1.0\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_is=true \
    algorithm.rollout_is_threshold=2.0 \
    algorithm.rollout_is_threshold_lower=0.0 \
    algorithm.rollout_is_level=sequence\
    algorithm.rollout_is_mode=truncate\
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb']\
    trainer.project_name='Pretrain-sharpen' \
    trainer.experiment_name=${model_name}-${dataset}-step60000-rollout${rollout_n}-lr${lr}-sam\
    reward_model.reward_manager=simple_math \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=false \
    trainer.nnodes=1 \
    trainer.save_freq=128 \
    trainer.test_freq=1230150121\
    trainer.total_training_steps=1024\
    trainer.total_epochs=25000 $@

done
done
done
done