#!/usr/bin/env bash
# On-policy distillation | text | vLLM rollout | FSDP training | NVIDIA GPUs

set -xeuo pipefail

# ---- user-adjustable ----
# STUDENT_MODEL="/data01/pengxia3/verl/models/Qwen2.5-Math-1.5B-EXT"
# TEACHER_MODEL="/data01/pengxia3/verl/models/Qwen2.5-Math-7B-EXT"
student_model="/data01/pengxia3/verl/models/Qwen3-4B-Instruct"
teacher_model="/data01/pengxia3/verl/models/Qwen3-8B-Instruct"

ngpus_per_node=4
teacher_world_size=4
nnodes=1

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

ppo_mini_batch_size=16
max_prompt_length=1024
max_response_length=7168
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=1
teacher_tp=1
rollout_gpu_mem_util=0.4
teacher_gpu_mem_util=0.4

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.4}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_distill_gsm8k_math}
exp_name=${EXPERIMENT_NAME:-qwen3_8b_from_qwen3_32b_vllm_fsdp}
# ---- end user-adjustable ----

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet

# train_files="['$gsm8k_train', '$math_train']"
# val_files="['$gsm8k_test', '$math_test']"
train_files="/data01/pengxia3/verl/data/gsm8k/train.parquet"
val_files="/data01/pengxia3/verl/data/gsm8k/test.parquet"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

trainer_args=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.validation_data_dir="${RAY_DATA_HOME}/val_data/${project_name}/${exp_name}"

    trainer.n_gpus_per_node=${ngpus_per_node}
    trainer.nnodes=${nnodes}

    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

data_args=(
    data.train_files=$train_files
    data.val_files=$val_files

    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
)

algorithm_args=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

model_args=(
    actor_rollout_ref.model.path=$student_model
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

actor_args=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

rollout_args=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

extra_args=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${teacher_world_size}
    distillation.nnodes=${nnodes}
    distillation.teacher_models.teacher_model.model_path=$teacher_model
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${trainer_args[@]}" \
    "${data_args[@]}" \
    "${algorithm_args[@]}" \
    "${model_args[@]}" \
    "${actor_args[@]}" \
    "${rollout_args[@]}" \
    "${extra_args[@]}" \
    "$@"
