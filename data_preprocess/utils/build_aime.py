import argparse
from functools import partial
import os
from datasets import load_dataset, Dataset
from .format import format_prompt, example_map_fn


def remove_boxed(s):
	if "\\boxed " in s:
		left = "\\boxed "
		assert s[: len(left)] == left
		return s[len(left) :]

	left = "\\boxed{"

	assert s[: len(left)] == left
	assert s[-1] == "}"

	return s[len(left) : -1]



def build_aime2024(enable_map=True):
	
	def process_aime2024(example):
		problem = example["problem"]
		ground_truth = example["solution"]
		ground_truth = remove_boxed(ground_truth)
		return problem, ground_truth

	data_source = "math-ai/aime24"
	dataset = load_dataset(data_source, split="test")
	map_fn = partial(
		example_map_fn, process_fn=process_aime2024, data_source=data_source, ability="math", split="test"
	)
	
	if enable_map:
		dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
	else:
		processed_data = []
		for idx, example in enumerate(dataset):
			processed_data.append(map_fn(example, idx))
		dataset = Dataset.from_list(processed_data)
	return dataset


def build_aime2025(enable_map=True):
	
	def process_aime2025(example):
		problem = example["problem"]
		ground_truth = example["answer"]
		return problem, ground_truth

	data_source = "math-ai/aime25"
	dataset = load_dataset(data_source, split="test")
	map_fn = partial(
		example_map_fn, process_fn=process_aime2025, data_source=data_source, ability="math", split="test"
	)

	if enable_map:
		dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
	else:
		processed_data = []
		for idx, example in enumerate(dataset):
			processed_data.append(map_fn(example, idx))
		dataset = Dataset.from_list(processed_data)
	return dataset


def build_aime2026(enable_map=True):
	
	def process_aime2026(example):
		problem = example["problem"]
		ground_truth = example["answer"]
		return problem, ground_truth

	data_source = "math-ai/aime26"
	dataset = load_dataset(data_source, split="test")
	map_fn = partial(
		example_map_fn, process_fn=process_aime2026, data_source=data_source, ability="math", split="test"
	)

	if enable_map:
		dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
	else:
		processed_data = []
		for idx, example in enumerate(dataset):
			processed_data.append(map_fn(example, idx))
		dataset = Dataset.from_list(processed_data)
	return dataset
