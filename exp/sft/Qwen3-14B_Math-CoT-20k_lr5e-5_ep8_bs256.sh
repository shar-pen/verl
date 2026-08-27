#!/usr/bin/env bash
set -xeuo pipefail

nproc_per_node=8

# ---- user-adjustable ----
# model
model_path="${RAY_DATA_HOME}/models/Qwen3/Qwen3-14B-Base"

# data
train_file="${RAY_DATA_HOME}/data/math_cot_20k/train.parquet"

# peft
use_peft=0
lora_rank=32
lora_alpha=16
lora_targets=all-linear

# training
sp_size=2
train_batch_size=256
max_length=32768
max_token_len_per_gpu=9216
lr=5e-5
total_steps=640
total_epochs=8
lr_scheduler_type=cosine
lr_warmup_ratio_percent=10
weight_decay=0.01

micro_batch_size_per_gpu=4
lr_warmup_steps=$(awk -v steps=${total_steps} -v percent=${lr_warmup_ratio_percent} 'BEGIN { printf "%d", steps * percent / 100 }')

# log
project_name="SFT_math_cot_20k"
experiment_name="Qwen3-14B_Math-CoT-20k_lr5e-5_ep8_bs256"
ckpts_dir="${RAY_DATA_HOME}/ckpts/${project_name}/${experiment_name}"

# ---- end user-adjustable ----

trainer_args=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}

    checkpoint.save_contents='["model","optimizer","extra","hf_model"]'

    trainer.save_freq=80
    trainer.total_training_steps=${total_steps}
    trainer.total_epochs=${total_epochs}

    trainer.default_local_dir=${ckpts_dir}
    trainer.resume_mode=disable
)

data_args=(
    data.train_batch_size=${train_batch_size}
    data.train_files=${train_file}
    data.micro_batch_size_per_gpu=${micro_batch_size_per_gpu}
    data.max_token_len_per_gpu=${max_token_len_per_gpu}

    data.messages_key=messages
    data.enable_thinking_key=enable_thinking
    data.shuffle=false
    data.custom_cls.path=pkg://verl.utils.dataset.last_turn_sft_dataset
    data.custom_cls.name=LastTurnSFTDataset

	data.max_length=${max_length}
	data.truncation=left
)

model_args=(
    model.path=${model_path}
    model.use_remove_padding=true
)

optim_args=(
    optim.lr=${lr}
	optim.lr_warmup_steps=${lr_warmup_steps}
    optim.lr_scheduler_type=${lr_scheduler_type}
	optim.weight_decay=${weight_decay}
    engine=fsdp
    engine.ulysses_sequence_parallel_size=${sp_size}
)

extra_args=()
if [ "${use_peft}" = "1" ]; then
    extra_args+=(
        model.lora_rank=${lora_rank}
        model.lora_alpha=${lora_alpha}
        model.target_modules=${lora_targets}
    )
fi


torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
    -m verl.trainer.sft_trainer \
    "${trainer_args[@]}" \
    "${data_args[@]}" \
    "${model_args[@]}" \
    "${optim_args[@]}" \
    "${extra_args[@]}" "$@"
