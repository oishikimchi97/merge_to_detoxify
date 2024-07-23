from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf

import wandb


def get_sweep_path(project_name: str, sweep_id: str) -> str:
    project_path = "/".join(wandb.Api().project(project_name).path)
    return f"{project_path}/{sweep_id}"


def get_split_dataset_path(cfg: OmegaConf) -> Path:
    dataset_base_dir = Path(cfg.dataset.dataset_base_dir)
    split = cfg.dataset.split
    return dataset_base_dir / split


def get_dir_paths(path: Path, match: str = None):
    dir_paths = path.glob("*/")
    dir_paths = [str(dir_path) for dir_path in dir_paths if dir_path.is_dir()]
    if match:
        dir_paths = [dir_path for dir_path in dir_paths if match in dir_path]
    return dir_paths


def get_hydra_config(config_name: str, overrides=None):
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
