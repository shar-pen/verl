
def format_prompt(problem: str):

	msgs = [
		{'role': 'user', 'content': problem},
	]
	return msgs


def example_map_fn(example, idx, process_fn, data_source, ability, split, process_fn_kwargs={}):
	problem, ground_truth = process_fn(example, **process_fn_kwargs)
	prompt = format_prompt(problem)
	data = {
		"data_source": data_source,
		"prompt": prompt,
		"ability": ability,
		"reward_model": {"style": "rule", "ground_truth": ground_truth},
		"extra_info": {"split": split, "index": idx, },
	}
	return data

def remove_boxed(s):
	if "\\boxed " in s:
		left = "\\boxed "
		assert s[: len(left)] == left
		return s[len(left) :]

	left = "\\boxed{"

	assert s[: len(left)] == left
	assert s[-1] == "}"

	return s[len(left) : -1]