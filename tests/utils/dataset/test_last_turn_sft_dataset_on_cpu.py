# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from verl.utils import hf_tokenizer
from verl.utils.dataset.last_turn_sft_dataset import LastTurnSFTDataset

default_model_path = Path("~/models/Qwen/Qwen3-0.6B").expanduser().resolve()


def _model_path():
    return Path(os.environ.get("LAST_TURN_SFT_TEST_MODEL", default_model_path)).resolve()


def _config():
    return {
        "messages_key": "messages",
        "max_length": 1024,
        "pad_mode": "no_padding",
        "truncation": "error",
    }


def test_last_turn_sft_dataset_preserves_final_reasoning(tmp_path):
    model_path = _model_path()
    if not model_path.exists():
        pytest.skip(f"Model is not available at {model_path}")

    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "reasoning_content": "historical reasoning", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "reasoning_content": "final reasoning", "content": "Second answer"},
    ]
    parquet_file = tmp_path / "last_turn.parquet"
    pd.DataFrame({"messages": [messages]}).to_parquet(parquet_file)

    tokenizer = hf_tokenizer(str(model_path))
    dataset = LastTurnSFTDataset(parquet_files=str(parquet_file), tokenizer=tokenizer, config=_config())
    item = dataset[0]

    expected_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True, return_tensors="pt"
    )[0]
    assert torch.equal(item["input_ids"], expected_ids)

    trained_text = tokenizer.decode(item["input_ids"][item["loss_mask"].bool()])
    context_text = tokenizer.decode(item["input_ids"][~item["loss_mask"].bool()])
    assert "historical reasoning" not in tokenizer.decode(item["input_ids"])
    assert "First answer" in context_text
    assert "Second question" in context_text
    assert "final reasoning" in trained_text
    assert "Second answer" in trained_text
    assert "final reasoning" not in context_text
    assert trained_text.startswith("<think>\n")


def test_last_turn_sft_dataset_rejects_non_assistant_final_message(tmp_path):
    model_path = _model_path()
    if not model_path.exists():
        pytest.skip(f"Model is not available at {model_path}")

    parquet_file = tmp_path / "invalid_last_turn.parquet"
    pd.DataFrame({"messages": [[{"role": "user", "content": "Question"}]]}).to_parquet(parquet_file)

    tokenizer = hf_tokenizer(str(model_path))
    dataset = LastTurnSFTDataset(parquet_files=str(parquet_file), tokenizer=tokenizer, config=_config())
    with pytest.raises(ValueError, match="end with an assistant"):
        dataset[0]


def test_last_turn_sft_dataset_padding_and_truncation(tmp_path):
    model_path = _model_path()
    if not model_path.exists():
        pytest.skip(f"Model is not available at {model_path}")

    messages = [
        {"role": "user", "content": "Question " * 20},
        {"role": "assistant", "reasoning_content": "Reasoning " * 20, "content": "Answer"},
    ]
    parquet_file = tmp_path / "length_handling.parquet"
    pd.DataFrame({"messages": [messages]}).to_parquet(parquet_file)
    tokenizer = hf_tokenizer(str(model_path))

    padded_config = _config() | {"pad_mode": "right", "max_length": 256}
    padded = LastTurnSFTDataset(parquet_files=str(parquet_file), tokenizer=tokenizer, config=padded_config)[0]
    assert padded["input_ids"].shape == padded["attention_mask"].shape == padded["loss_mask"].shape == (256,)
    assert torch.all(padded["loss_mask"][padded["attention_mask"] == 0] == 0)

    full_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    truncated_config = _config() | {"max_length": len(full_ids) - 5, "truncation": "left"}
    truncated = LastTurnSFTDataset(parquet_files=str(parquet_file), tokenizer=tokenizer, config=truncated_config)[0]
    assert truncated["input_ids"].tolist() == full_ids[-truncated_config["max_length"] :]
