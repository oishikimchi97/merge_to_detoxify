import os
from pathlib import Path

import hydra
import torch
import tqdm
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, set_seed

from mtd.task_vector.task_vector import TaskVector
from mtd.utils.config import get_pretrained_model_name


@hydra.main(
    version_base=None, config_path="../config", config_name="cal_task_vector_minmax"
)
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "false" if cfg.wandb else "true"

    model_path = Path(cfg.model_path)
    pretrained_path = Path(get_pretrained_model_name(model_path))

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    print(f"Model loaded from {model_path}")
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        pretrained_path, device_map="auto"
    )
    print(f"Pretrained model loaded from {pretrained_path}")

    task_vector_state_dict = TaskVector(pretrained_model, model).vector
    task_vector_params = torch.cat(
        [
            para.flatten()
            for para in tqdm.tqdm(
                task_vector_state_dict.values(),
                desc=f"Calculating the task vector min_max for {model_path}",
            )
        ]
    )
    task_vector_min, task_vector_max = (
        torch.max(task_vector_params).cpu().item(),
        torch.min(task_vector_params).cpu().item(),
    )
    print(f"Task vector max: {task_vector_max}, min: {task_vector_min}")

    if cfg.wandb:
        import wandb

        run = wandb.init(project="cal_task_vector_minmax")
        run_name = "-".join(cfg.model_path.split("/")[1:])
        run.name = run_name
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

        wandb.log({"min": task_vector_min, "max": task_vector_max})

        wandb.finish()


if __name__ == "__main__":
    main()
