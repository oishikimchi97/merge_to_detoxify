import argparse

import wandb
from utils import get_hydra_config, get_sweep_path

RANDOM_SEEDS = [42, 84, 126, 168, 210]
DEFAULT_SEED = [42]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="phi-1_5", type=str, help="The name of the model"
    )
    parser.add_argument(
        "--multi_gpu",
        default=2,
        type=int,
        help="The number of GPUs to use for per train",
    )
    parser.add_argument(
        "--dataset",
        default="train/toxic_civil_comments",
        type=str,
        help="Dataset name to use",
    )
    parser.add_argument(
        "--trainer",
        default="paged_adamw-descent",
        type=str,
        help="The name of the trainer",
    )
    args = parser.parse_args()

    model = args.model
    dataset = args.dataset
    multi_gpu_num = args.multi_gpu
    trainer = args.trainer

    overrides = [f"model={model}", f"dataset={dataset}", f"trainer={trainer}"]
    cfg = get_hydra_config("train", overrides=overrides)

    command = [
        "${env}",
        "torchrun",
        f"--nproc_per_node={multi_gpu_num}",
        "${program}",
        "${args_no_hyphens}",
    ]

    sweep_config = {
        "program": "train_with_multi_gpus.py",
        "method": "grid",
        "parameters": {
            "model": {"value": model},
            "dataset": {"value": dataset},
            "trainer": {"value": trainer},
        },
        "command": command,
    }

    project_name = cfg.project_name

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    sweep_path = get_sweep_path(project_name, sweep_id)
    print(sweep_path)
