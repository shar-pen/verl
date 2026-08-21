from datasets import Features, List, Value, load_dataset

SFT_FEATURES = Features(
    {
        "enable_thinking": Value("bool"),
        "messages": List(
            {
                "role": Value("string"),
                "content": Value("string"),
                "reasoning_content": Value("string"),
            }
        )
    }
)


def split_reasoning_response(response: str) -> tuple[str, str]:
    think_start = "<think>"
    think_end = "</think>"

    if not response.startswith(think_start):
        raise ValueError("response must start with <think>")
    if think_end not in response:
        raise ValueError("response is missing </think>")

    reasoning_content, content = response[len(think_start) :].split(think_end, maxsplit=1)
    reasoning_content = reasoning_content.strip()
    content = content.strip()

    if not reasoning_content:
        raise ValueError("reasoning_content must not be empty")
    if not content:
        raise ValueError("assistant content must not be empty")

    return reasoning_content, content


def process_fn(example):
    reasoning_content, content = split_reasoning_response(example["response"])
    assistant_message = {
        "role": "assistant",
        "reasoning_content": reasoning_content,
        "content": content,
    }
    return {
        "enable_thinking": True,
        "messages": example["message"] + [assistant_message],
    }


def build_math_cot_20k():
    dataset = load_dataset("jasonrqh/Math-CoT-20k")
    train_dataset = dataset["train"]
    return train_dataset.map(
        process_fn,
        remove_columns=train_dataset.column_names,
        features=SFT_FEATURES,
        load_from_cache_file=False,
    )
