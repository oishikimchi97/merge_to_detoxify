import json
from pathlib import Path


def load_train_config(path) -> dict:
    if path is str:
        path = Path(path)

    with open(path, "r") as file:
        config = json.load(file)
    return config
