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

"""SFT dataset that trains only the final assistant turn of a conversation."""

from typing import Any, Optional

import torch
import torch.nn.functional as F

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.chat_template import apply_chat_template
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class LastTurnSFTDataset(MultiTurnSFTDataset):
    """Render the full conversation and apply loss only after the final generation prompt.

    Each sample must be a conversation prefix ending in an assistant message. For
    multi-turn training, split a conversation into one prefix per assistant turn.
    """

    def _tokenize_last_turn(
        self, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]], enable_thinking: Optional[bool]
    ):
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("LastTurnSFTDataset requires every conversation to end with an assistant message")
        if not any(message["role"] == "user" for message in messages[:-1]):
            raise ValueError("LastTurnSFTDataset requires a user message before the final assistant message")

        processor = self.processor if self.processor is not None else self.tokenizer
        full_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            full_template_kwargs["enable_thinking"] = enable_thinking

        full_inputs = apply_chat_template(
            processor,
            messages=messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **full_template_kwargs,
        )

        # enable_thinking controls generated response prefixes. It must not insert
        # an empty thinking block when measuring the start of an existing response.
        prompt_template_kwargs = {
            key: value for key, value in full_template_kwargs.items() if key != "enable_thinking"
        }
        prompt_inputs = apply_chat_template(
            processor,
            messages=messages[:-1],
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **prompt_template_kwargs,
        )

        full_inputs = dict(full_inputs)
        input_ids = full_inputs.pop("input_ids")[0]
        attention_mask = full_inputs.pop("attention_mask")[0]
        prompt_ids = prompt_inputs["input_ids"][0]

        if len(prompt_ids) >= len(input_ids) or not torch.equal(input_ids[: len(prompt_ids)], prompt_ids):
            raise ValueError(
                "The generation prompt is not a strict prefix of the rendered conversation; "
                "the final assistant loss boundary cannot be determined safely"
            )

        loss_mask = attention_mask.clone()
        loss_mask[: len(prompt_ids)] = 0
        full_inputs.pop("mm_token_type_ids", None)
        return input_ids, loss_mask, attention_mask, full_inputs

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        input_ids, loss_mask, attention_mask, multi_modal_inputs = self._tokenize_last_turn(
            messages, tools, enable_thinking
        )
        assert input_ids.shape == loss_mask.shape == attention_mask.shape
        self.sanity_check(input_ids, messages, tools, enable_thinking)

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                second_per_grid_ts=multi_modal_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            if self.truncation == "left":
                sequence_slice = slice(-self.max_length, None)
            elif self.truncation == "right":
                sequence_slice = slice(None, self.max_length)
            elif self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")
            input_ids = input_ids[sequence_slice]
            attention_mask = attention_mask[sequence_slice]
            loss_mask = loss_mask[sequence_slice]
            position_ids = position_ids[..., sequence_slice]
            sequence_length = input_ids.shape[0]

        if self.pad_mode == DatasetPadMode.RIGHT:
            pad_length = self.max_length - sequence_length
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                input_ids = F.pad(input_ids, (0, pad_length), value=pad_token_id)
                attention_mask = F.pad(attention_mask, (0, pad_length), value=0)
                loss_mask = F.pad(loss_mask, (0, pad_length), value=0)
                position_ids = F.pad(position_ids, (0, pad_length), value=0)
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            result = {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

        if multi_modal_inputs:
            result["multi_modal_inputs"] = multi_modal_inputs
        return result
