import argparse

import wandb
from utils import get_hydra_config, get_sweep_path

RANDOM_SEEDS = [42, 84, 126, 168, 210]
DEFAULT_SEED = [42]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gpt2", type=str, help="The name of the model"
    )
    parser.add_argument(
        "--num_gpu",
        default=1,
        type=int,
        help="The number of GPUs to use for per train",
    )
    parser.add_argument(
        "--dataset",
        default="",
        type=str,
        help="Path to the directory",
    )
    parser.add_argument(
        "--trainer", default="", type=str, help="The name of the trainer"
    )
    args = parser.parse_args()

    model = args.model
    num_gpu = args.num_gpu
    dataset = args.dataset
    trainer = args.trainer
    if trainer:
        overrides = [f"model={model}", f"dataset={dataset}", f"model/trainer={trainer}"]
    else:
        overrides = [f"model={model}", f"dataset={dataset}"]
    cfg = get_hydra_config("train", overrides=overrides)

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

    sweep_config = {
        "program": "scripts/train.py",
        "method": "grid",
        "parameters": {
            "model": {"value": model},
            "dataset": {"value": dataset},
        },
        "command": command,
    }
    if trainer:
        sweep_config["parameters"]["model/trainer"] = {"value": trainer}

    project_name = cfg.project_name

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    sweep_path = get_sweep_path(project_name, sweep_id)
    print(sweep_path)
