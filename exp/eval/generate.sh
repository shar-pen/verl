#!/usr/bin/env bash

# Rollout-only generation for offline evaluation.
# The output parquet keeps the original dataset columns and adds ``responses``.

set -xeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

model_path="${RAY_DATA_HOME}/ckpts/SFT_math_cot_20k/qwen3_14b_32k/global_step_160_merged_hf_model"
train_file="${RAY_DATA_HOME}/data/aime/aime2024.parquet"
output_path="${script_dir}/outputs/aime2024_qwen3_14bsft_step_160.parquet"

nnodes=1
ngpus_per_node=8
prompt_length=$((1024 * 2))
response_length=$((1024 * 30))
# Limit vLLM's KV-cache reservation. Override this for smaller-memory GPUs,
# e.g. ``max_model_len=8192``. It must be greater than prompt_length.
max_model_len=${max_model_len:-$((prompt_length + response_length + 1))}
max_num_batched_tokens=${max_num_batched_tokens:-${max_model_len}}

tp=${tp:-1}
rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.9}
n_samples=${n_samples:-1}
temperature=${temperature:-1.0}
top_p=${top_p:-0.7}
top_k=${top_k:--1}

if (( max_model_len <= prompt_length )); then
    echo "max_model_len (${max_model_len}) must be greater than prompt_length (${prompt_length})" >&2
    exit 1
fi

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${ngpus_per_node}" \
    data.train_files="${train_file}" \
    data.prompt_key=prompt \
    +data.output_path="${output_path}" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.prompt_length="${prompt_length}" \
    actor_rollout_ref.rollout.response_length="${response_length}" \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${max_num_batched_tokens}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${tp}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_mem_util}" \
    actor_rollout_ref.rollout.temperature="${temperature}" \
    actor_rollout_ref.rollout.top_p="${top_p}" \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.n="${n_samples}" \
    "$@"
