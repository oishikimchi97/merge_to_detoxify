import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, set_seed

from mtd.eval.task_vector_norm import cal_task_vector_norm
from mtd.utils.config import get_pretrained_model_name


@hydra.main(
    version_base=None, config_path="../config", config_name="cal_task_vector_norm"
)
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "false" if cfg.wandb else "true"

    model_path = Path(cfg.model_path)
    pretrained_path = Path(get_pretrained_model_name(model_path))

    model = AutoModelForCausalLM.from_pretrained(model_path)
    print(f"Model loaded from {model_path}")
    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_path)
    print(f"Pretrained model loaded from {pretrained_path}")

    task_vector_norm = cal_task_vector_norm(model, pretrained_model, cfg.p)
    print(f"Task vector norm: {task_vector_norm}")

    if cfg.wandb:
        import wandb

        run = wandb.init(project="cal_task_vector_norm")
        run_name = "-".join(cfg.model_path.split("/")[1:])
        run.name = run_name
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

        wandb.log({"task_vector_norm": task_vector_norm})

        wandb.finish()


if __name__ == "__main__":
    main()
