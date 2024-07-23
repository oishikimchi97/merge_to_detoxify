from pathlib import Path

from mtd.utils import PATH_TYPE
from mtd.utils.config import read_model_config


def get_negated_dir(finetune_path: PATH_TYPE, coef: float) -> Path:
    if not isinstance(finetune_path, Path):
        finetune_path = Path(finetune_path)
    return finetune_path.parent / f"{finetune_path.name}-negated-{coef:.2f}"


def get_pretrained_path(finetune_path: PATH_TYPE) -> Path:
    config = read_model_config(finetune_path)
    pretrained_path = config["_name_or_path"]
    return pretrained_path
