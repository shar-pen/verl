# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline evaluation of generated responses."""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.trainer.ppo.metric_utils import (
    compute_validation_response_length_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score


def _as_list(value):
    if isinstance(value, str) or not isinstance(value, (list, tuple, np.ndarray)):
        return [value]
    return list(value)


@ray.remote
def process_item(config, uid, data_source, response_lst, reward_data, response_lengths=None):
    """Score every response for one prompt and retain reward components."""
    reward_fn = get_custom_reward_fn(config) or default_compute_score
    ground_truth = reward_data["ground_truth"]
    response_lst = _as_list(response_lst)
    response_lengths = [None] * len(response_lst) if response_lengths is None else _as_list(response_lengths)
    if len(response_lengths) != len(response_lst):
        raise ValueError(
            f"response_lengths must match responses for uid {uid}: "
            f"got {len(response_lengths)} lengths and {len(response_lst)} responses"
        )

    rewards = []
    reward_extra_infos = []
    for response in response_lst:
        score = reward_fn(data_source, response, ground_truth)
        if isinstance(score, dict):
            if "score" not in score and "acc" not in score:
                raise ValueError(f"Reward result for uid {uid} must contain 'score' or 'acc': {score}")
            reward = score.get("score", score["acc"])
            reward_extra_info = dict(score)
        else:
            reward = score
            # RL reward-loop managers expose scalar verifier results as acc.
            reward_extra_info = {"acc": float(score)}
        rewards.append(float(reward))
        reward_extra_infos.append(reward_extra_info)

    return {
        "uid": uid,
        "data_source": data_source,
        "rewards": rewards,
        "reward_extra_infos": reward_extra_infos,
        "response_lengths": response_lengths,
    }


def _aggregate_rl_style_metrics(results, max_response_length=None):
    """Aggregate offline results using the normal RL validation metric logic."""
    records_by_data_source = defaultdict(list)
    response_lengths_by_data_source = defaultdict(list)

    for result in results:
        uid = result["uid"]
        data_source = result["data_source"]
        for reward, extra_info, response_length in zip(
            result["rewards"],
            result["reward_extra_infos"],
            result["response_lengths"],
            strict=True,
        ):
            records_by_data_source[data_source].append({"uid": uid, "reward": reward, **extra_info})
            if response_length is not None:
                response_lengths_by_data_source[data_source].append(response_length)

    metric_dict = {}
    for data_source, records in records_by_data_source.items():
        sample_uids = [record["uid"] for record in records]
        common_keys = set.intersection(*(set(record) for record in records)) - {"uid"}
        infos_dict = {
            key: [record[key] for record in records]
            for key in sorted(common_keys)
            if isinstance(records[0][key], (int, float, bool, np.number, str))
        }
        processed = process_validation_metrics(
            data_sources=[data_source] * len(records),
            sample_uids=sample_uids,
            infos_dict=infos_dict,
        )[data_source]
        core_var = "acc" if "acc" in processed else "reward"
        for var_name, metric2val in processed.items():
            n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val)
            for metric_name, metric_val in metric2val.items():
                is_core = (
                    var_name == core_var
                    and metric_name.startswith(("mean", "maj", "best", "max"))
                    and f"@{n_max}" in metric_name
                )
                section = "val-core" if is_core else "val-aux"
                metric_dict[f"{section}/{data_source}/{var_name}/{metric_name}"] = float(metric_val)

        # Keep the original standalone metric for existing result parsers.
        metric_dict[f"test_score/{data_source}"] = float(np.mean([record["reward"] for record in records]))

    length_data_sources = []
    response_lengths = []
    for data_source, lengths in response_lengths_by_data_source.items():
        length_data_sources.extend([data_source] * len(lengths))
        response_lengths.extend(lengths)

    if response_lengths:
        length_metrics = compute_validation_response_length_metrics(
            data_sources=length_data_sources,
            response_lengths=response_lengths,
            max_response_length=None if max_response_length is None else int(max_response_length),
        )
        for data_source, metrics in length_metrics.items():
            for metric_name, metric_val in metrics.items():
                metric_dict[f"val-aux/{data_source}/{metric_name}"] = float(metric_val)

                # Keep the previous main_eval response-length key layout.
                prefix, leaf = metric_name.rsplit("/", 1)
                metric_dict[f"test_{prefix}/{data_source}/{leaf}"] = float(metric_val)

    return metric_dict


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    response_lengths_key = config.data.get("response_lengths_key", "response_lengths")
    response_lengths_data = dataset[response_lengths_key] if response_lengths_key in dataset else None
    uid_key = config.data.get("uid_key", "uid")
    sample_uids = dataset[uid_key].astype(str).tolist() if uid_key in dataset else [str(i) for i in range(len(dataset))]

    if not ray.is_initialized():
        ray.init(**OmegaConf.to_container(config.ray_kwargs.get("ray_init", {})))

    remote_tasks = [
        process_item.remote(
            config,
            sample_uids[i],
            data_sources[i],
            responses[i],
            reward_model_data[i],
            None if response_lengths_data is None else response_lengths_data[i],
        )
        for i in range(len(dataset))
    ]

    results = []
    with tqdm(total=len(dataset)) as pbar:
        while remote_tasks:
            done_ids, remote_tasks = ray.wait(remote_tasks)
            results.extend(ray.get(done_ids))
            pbar.update(len(done_ids))

    metric_dict = _aggregate_rl_style_metrics(
        results=results,
        max_response_length=config.data.get("max_response_length", None),
    )
    print(metric_dict)


if __name__ == "__main__":
    main()
