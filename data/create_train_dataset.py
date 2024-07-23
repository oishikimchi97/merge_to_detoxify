from pathlib import Path

import hydra
from omegaconf import OmegaConf

import datasets


# TODO: Change this code to be available other type of datasets (e.g Toxigen, non-toxic type datasets)
@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="create_train_dataset",
)
def main(cfg: OmegaConf):
    output_dir = Path(f"{cfg.dataset.dataset_base_dir}/all")
    if output_dir.exists():
        print(f"Dataset already exists at {cfg.dataset.dataset_base_dir}")
        return
    dataset = datasets.load_dataset(
        cfg.dataset.hf_key,
        name=cfg.dataset.hf_subset if cfg.dataset.hf_subset else None,
    )
    for split in cfg.dataset.hf_splits:
        dataset[split] = dataset[split].filter(
            get_toxic_filter_func(
                cfg.dataset.toxic_label,
                cfg.dataset.toxicity_threshold,
                cfg.dataset.toxic_type,
            )
        )
    dataset.save_to_disk(output_dir)
    print(f"Saved dataset to {output_dir}")


def get_toxic_filter_func(
    toxic_label: str, toxicity_threshold: float, toxic_type: str = "toxic"
):
    assert toxic_type in [
        "toxic",
        "non-toxic",
    ], "toxic_type must be either 'toxic' or 'non-toxic'"
    if toxic_type == "toxic":
        return lambda x: x[toxic_label] > toxicity_threshold
    elif toxic_type == "non-toxic":
        return lambda x: x[toxic_label] < toxicity_threshold


if __name__ == "__main__":
    main()
