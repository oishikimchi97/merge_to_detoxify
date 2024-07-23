from pathlib import Path

import hydra
from datasets.dataset_dict import DatasetDict
from omegaconf import OmegaConf
from transformers import set_seed

import datasets


def sample_dataset_by_ratio(dataset: DatasetDict, sample_ratio: float) -> DatasetDict:
    for split_name in dataset:
        entity = dataset[split_name]
        sample_num = int(len(entity) * sample_ratio)
        sampled_entity = entity.shuffle().select(range(sample_num))
        dataset[split_name] = sampled_entity
    return dataset


@hydra.main(
    version_base=None, config_path="../config", config_name="sample_dataset_by_ratio"
)
def main(cfg: OmegaConf):
    assert cfg.dataset.sample_ratio > 0, "please set sample_ratio > 0"
    set_seed(cfg.seed)

    dataset_path = Path(cfg.dataset.dataset_path)
    sample_ratio = cfg.dataset.sample_ratio

    dataset = datasets.load_from_disk(dataset_path)
    sampled_dataset = sample_dataset_by_ratio(dataset, sample_ratio)

    output_path = dataset_path.parent / f"sampled_{sample_ratio:.2f}"
    sampled_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    main()
