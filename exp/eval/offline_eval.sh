#!/usr/bin/env bash

# Offline scoring for a parquet produced by generate.sh or main_generation_server.

set -xeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
generated_path=${generated_path:-${script_dir}/outputs/aime2024_qwen3_14bsft_step_800.parquet}
response_length=${response_length:-$((1024 * 30))}

python3 -m verl.trainer.main_eval \
    data.path="${generated_path}" \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model \
    +data.response_lengths_key=response_lengths \
    +data.max_response_length="${response_length}" \
    "$@"
