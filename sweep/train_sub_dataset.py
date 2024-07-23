import argparse
from pathlib import Path

import wandb
from utils import (
    get_dir_paths,
    get_hydra_config,
    get_split_dataset_path,
    get_sweep_path,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="default", type=str, help="The name of the model"
    )
    parser.add_argument(
        "--num_gpu",
        default=1,
        type=int,
        help="The number of GPUs to use for per train",
    )
    parser.add_argument(
        "--dataset",
        default="train/toxic_civil_comments",
        type=str,
        help="Path to the directory",
    )
    parser.add_argument(
        "--split", default="5", type=str, help="The number of the dataset split"
    )
    parser.add_argument(
        "--trainer", default="", type=str, help="The name of the trainer"
    )
    args = parser.parse_args()

    model = args.model
    num_gpu = args.num_gpu
    dataset = args.dataset
    split_name = f"split_{args.split}"
    trainer = args.trainer
    if num_gpu > 1:
        command = [
            "${env}",
            "torchrun",
            f"--nproc_per_node={num_gpu}",
            "${program}",
            "${args_no_hyphens}",
        ]
    else:
        command = ["${env}", "python3", "${program}", "${args_no_hyphens}"]
    if trainer:
        overrides = [
            f"model={model}",
            f"dataset={dataset}",
            f"dataset.split={split_name}",
            f"model/trainer={trainer}",
        ]
    else:
        overrides = [
            f"model={model}",
            f"dataset={dataset}",
            f"dataset.split={split_name}",
        ]
    cfg = get_hydra_config("train", overrides=overrides)
    project_name = cfg.project_name

    split_dataset_path = get_split_dataset_path(cfg)
    sub_dirs = get_dir_paths(split_dataset_path)
    sub_dir_names = [str(Path(sub_dir).name) for sub_dir in sub_dirs]

    sweep_config = {
        "program": "scripts/train.py",
        "method": "grid",
        "parameters": {
            "model": {"value": model},
            "dataset": {"value": dataset},
            "dataset.split": {"value": split_name},
            "dataset.sub_dir_name": {"values": sub_dir_names},
        },
        "command": command,
    }
    if trainer:
        sweep_config["parameters"]["model/trainer"] = {"value": trainer}

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    sweep_path = get_sweep_path(project_name, sweep_id)
    print(sweep_path)
