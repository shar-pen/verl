#!/usr/bin/env bash

# Generate responses and then score the generated parquet.

set -xeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_path=${output_path:-${script_dir}/outputs/aime2024_qwen3_14bsft_step_160.parquet}

output_path="${output_path}" "${script_dir}/generate.sh" "$@"
generated_path="${output_path}" "${script_dir}/offline_eval.sh"
