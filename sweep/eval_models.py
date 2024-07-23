import argparse
from pathlib import Path

import wandb
from utils import get_dir_paths, get_hydra_config, get_sweep_path


def get_eval_project_name(model_dir, toxicity_name: str, prompt_name: str) -> str:
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)
    toxicity_suffix = f"-{toxicity_name}-{prompt_name}-eval"
    model_name = str(model_dir.parent.name)
    experiment_name = str(model_dir.name)
    project_name = f"{model_name}-{experiment_name}"

    if "-train" in project_name:
        project_name = project_name.replace("-train", toxicity_suffix)
    else:
        project_name = project_name + toxicity_suffix

    return project_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to the directory")
    parser.add_argument(
        "--toxic_type",
        default="detoxify",
        type=str,
        choices=["detoxify", "toxigen_hatebert"],
        help="Path to the toxic data directory",
    )
    parser.add_argument(
        "--match",
        default=None,
        type=str,
        help="Match the directory name with the given string. If not provided, all the directories will be used.",
    )
    args = parser.parse_args()

    dir_path = Path(args.model_dir)
    toxic_type = args.toxic_type

    overrides = [f"eval/toxicity={toxic_type}"]
    cfg = get_hydra_config("eval", overrides=overrides)

    project_name = get_eval_project_name(
        dir_path, cfg.eval.toxicity.name, cfg.eval.toxicity.prompt.name
    )

    dir_paths = get_dir_paths(dir_path, match=args.match)

    sweep_config = {
        "program": "eval.py",
        "method": "grid",
        "metric": {"goal": "minimize", "name": "toxicity avg_score"},
        "parameters": {
            "model_path": {"values": dir_paths},
            "eval/toxicity": {"value": toxic_type},
        },
        "command": ["${env}", "python3", "${program}", "${args_no_hyphens}"],
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    sweep_path = get_sweep_path(project_name, sweep_id)
    print(sweep_path)
