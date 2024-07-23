import argparse
from pathlib import Path

import wandb
from utils import get_dir_paths, get_sweep_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to the directory")
    parser.add_argument(
        "--match",
        default=None,
        type=str,
        help="Match the directory name with the given string. If not provided, all the directories will be used.",
    )
    args = parser.parse_args()

    dir_path = Path(args.model_dir)

    dir_paths = get_dir_paths(dir_path, match=args.match)

    sweep_config = {
        "program": "get_task_vector_histogram.py",
        "method": "grid",
        "parameters": {
            "model_path": {"values": dir_paths},
        },
        "command": ["${env}", "python3", "${program}", "${args_no_hyphens}"],
    }

    project_name = "eval_task_vector_hist"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    sweep_path = get_sweep_path(project_name, sweep_id)
    print(sweep_path)
