from pathlib import Path
from typing import List

import hydra
import numpy as np
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from omegaconf import OmegaConf
from transformers import set_seed

import datasets


def split_dataset(dataset: DatasetDict, split_num: int = 5) -> List[DatasetDict]:
    assert isinstance(dataset, DatasetDict), print("dataset must be a DatasetDict")
    dataset_keys = list(dataset.keys())
    entities = [dataset[key] for key in dataset_keys]
    sub_entities = []
    for entity in entities:
        assert isinstance(entity, Dataset), print("entity must be a Dataset")
        entity_size = len(entity)
        index_arr = np.arange(entity_size)
        np.random.shuffle(index_arr)

        sub_indices = np.array_split(index_arr, split_num)
        sub_entity = [entity.select(sub_index) for sub_index in sub_indices]
        sub_entities.append(sub_entity)

    sub_datasets = []
    for sub_dataset in zip(*sub_entities):
        sub_dataset_dict = dict(zip(dataset_keys, sub_dataset))
        sub_datasets.append(DatasetDict(sub_dataset_dict))

    return sub_datasets


@hydra.main(version_base=None, config_path="../config", config_name="split_dataset")
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    print(f"Random seed: {cfg.seed}")

    dataset_base_dir = Path(cfg.dataset.dataset_base_dir)
    split_num = cfg.split

    dataset_parent_path = dataset_base_dir
    sub_parent_path = dataset_parent_path / f"{cfg.split_dataset_prefix}_{split_num}"
    is_exist = sub_parent_path.exists()

    if is_exist:
        print(f"{sub_parent_path} already exists")
        print("Dataset splitting is not performed")
        print("If you want to perform splitting, please remove the directory first")
    else:
        dataset = datasets.load_from_disk(str(dataset_base_dir / "all"))

        sub_datasets = split_dataset(dataset, split_num=split_num)

        for i, sub_dataset in enumerate(sub_datasets):
            sub_dataset_path = sub_parent_path / f"{cfg.sub_dataset_prefix}{i}"

            sub_dataset.save_to_disk(sub_dataset_path)
            print(f"Saved to {sub_dataset_path}")


if __name__ == "__main__":
    main()
