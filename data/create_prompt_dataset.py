from pathlib import Path

import datasets
import hydra
from datasets.dataset_dict import DatasetDict
from omegaconf import OmegaConf
from transformers import set_seed


def sample_dataset_by_num(dataset: DatasetDict, sample_num: int) -> DatasetDict:
    for split_name in dataset:
        entity = dataset[split_name]
        sampled_entity = entity.shuffle().select(range(sample_num))
        dataset[split_name] = sampled_entity
    return dataset


@hydra.main(
    version_base=None, config_path="../config", config_name="create_prompt_dataset"
)
def main(cfg: OmegaConf):
    assert cfg.dataset.sample_num > 0, "please set sample_ratio > 0"
    set_seed(cfg.seed)

    dataset = datasets.load_dataset(cfg.dataset.hf_key, name=cfg.dataset.hf_sub_set)
    output_path = Path(cfg.dataset.dataset_path)
    sample_num = cfg.dataset.sample_num

    prompt_label = cfg.dataset.prompt_label
    label = cfg.dataset.label

    filtered_dataset = dataset.filter(lambda x: x[prompt_label] == label)
    sampled_dataset = sample_dataset_by_num(filtered_dataset, sample_num)

    output_path = output_path.parent / f"sampled_{sample_num}"
    sampled_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    main()
