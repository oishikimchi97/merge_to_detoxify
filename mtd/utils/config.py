import json
from pathlib import Path

from mtd.utils import PATH_TYPE


def read_train_config(path) -> dict:
    if path is str:
        path = Path(path)

    with open(path, "r") as file:
        config = json.load(file)
    return config


def read_model_config(model_path: PATH_TYPE) -> dict:
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    config_file = model_path / "config.json"
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def get_pretrained_model_name(model_path: PATH_TYPE) -> str:
    return read_model_config(model_path)["_name_or_path"]
