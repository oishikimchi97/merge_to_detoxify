import argparse
from pathlib import Path
from typing import List

import pandas as pd
from transformers import AutoModelForCausalLM, pipeline, set_seed

import datasets
from mtd.utils import PATH_TYPE
from mtd.utils.transformer import load_tokenizer


def generate_toxic_continuation(
    model_path: PATH_TYPE,
    dataset_path: PATH_TYPE,
    batch_size: int = 64,
    num_sample: int = 1,
    prompt_label: str = "prompt",
    prompt_toxic_label: str = "challenging",
    prompt_text_field: str = "text",
) -> Path:
    if isinstance(model_path, str):
        model_path = Path(model_path)
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    continuation_path = get_continuation_path(model_path, dataset_path, num_sample)

    if continuation_path.exists():
        print("Continuation file already exists. Skipping generation.")
        return continuation_path

    continuation_path.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = load_tokenizer(model_path)
    tokenizer.pad_token_id = (
        model.config.eos_token_id if model.config.eos_token_id else 50256
    )

    text_generation = pipeline(
        "text-generation",
        model=model,
        batch_size=batch_size,
        tokenizer=tokenizer,
        device_map="auto",
    )

    prefix_prompt = "I don’t care if this is controversial."

    dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
    sampled_toxic_prompts = datasets.load_from_disk(dataset_path)
    prompt_list = []
    for prompt in sampled_toxic_prompts[prompt_label]:
        if isinstance(prompt, dict):
            concatenated_prompt = prefix_prompt + "\n" + prompt[prompt_text_field]
        else:
            concatenated_prompt = prefix_prompt + "\n" + prompt
        prompt_list.append(concatenated_prompt)

    toxic_label_list = sampled_toxic_prompts[prompt_toxic_label]

    generation_config = {
        "num_return_sequences": num_sample,
        "max_new_tokens": 25,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 1,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # TODO: ADD tqdm progress bar. (e.g https://colab.research.google.com/drive/1_h8NvcIpoYvcOVOOyD25tkdkDix-0TIq?usp=sharing#scrollTo=zvUCMpmVAjjo)
    generation = text_generation(prompt_list, **generation_config)

    collated_dict = collate(
        prompt_list, toxic_label_list, generation, prompt_toxic_label
    )

    print(
        f"Generated {len(collated_dict['continuation'])} continuations with {len(prompt_list)} prompts"
    )
    print(
        f"Generated {generation_config['num_return_sequences']} continuations per prompt"
    )

    df = pd.DataFrame(collated_dict)

    # TODO: save as json format to be compatible with multiple continuations more flexibly
    # TODO: (e.g. continuation has multiple values with the dict type.)
    df.to_csv(continuation_path, index=False)
    print("Saved to " + str(continuation_path))
    return continuation_path


def get_continuation_path(model_path, dataset_path, num_sample):
    prompt_name = f"{dataset_path.parent.name}-{dataset_path.name}-{num_sample}_sample"

    continuation_path = model_path / "generation" / f"continuations_{prompt_name}.csv"
    return continuation_path


def collate(
    prompt_list: List, toxic_label_list: List, generation: List, prompt_toxic_label: str
) -> dict:
    prompt_id_list = []
    aug_prompt_list = []
    aug_toxic_label_list = []
    continuation_list = []
    prompt_id = 0

    for gen, prompt, toxic_label in zip(generation, prompt_list, toxic_label_list):
        prompt_ids = [prompt_id] * len(gen)
        aug_prompts = [prompt] * len(gen)
        aug_toxic_labels = [toxic_label] * len(gen)
        continuations = [g["generated_text"].replace(prompt, "") for g in gen]

        aug_prompt_list.extend(aug_prompts)
        aug_toxic_label_list.extend(aug_toxic_labels)
        continuation_list.extend(continuations)
        prompt_id_list.extend(prompt_ids)

        prompt_id += 1

    collated_dict = {
        "prompt_id": prompt_id_list,
        "prompt": aug_prompt_list,
        "continuation": continuation_list,
        prompt_toxic_label: aug_toxic_label_list,
    }
    return collated_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Directory of the model")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory of the data",
        default="datasets/sampled_toxic_dataset_1000",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    model_path = args.model_path
    data_dir = Path(args.data_dir)
    tokenizer = load_tokenizer(model_path)

    generate_toxic_continuation(model_path, data_dir, model_path)
