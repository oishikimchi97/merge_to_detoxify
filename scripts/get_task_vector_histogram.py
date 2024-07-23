import os
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, set_seed

from mtd.task_vector.task_vector import TaskVector
from mtd.utils.config import get_pretrained_model_name


@hydra.main(
    version_base=None, config_path="../config", config_name="get_task_vector_histogram"
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
                desc="flatten task vector parameters",
            )
        ]
    )
    task_vector_params = torch.abs(task_vector_params).cpu().numpy()

    counts, bin_edges = np.histogram(
        task_vector_params, bins=int(cfg.num_bin), range=(0, cfg.bin_max)
    )

    output_path = model_path / "task_vector_histogram"
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "counts.npy", counts)
    np.save(output_path / "bin_edges.npy", bin_edges)

    print(f"Task vector histogram saved to {output_path}")

    if cfg.wandb:
        import wandb

        run = wandb.init(project="task_vector_histogram")
        run_name = "-".join(cfg.model_path.split("/")[1:])
        run.name = run_name
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        hist = wandb.Histogram(np_histogram=(counts, bin_edges))
        fig, ax = plt.subplots()
        ax.stairs(counts, bin_edges, fill=True)

        wandb.log({"task_vector_histogram": hist})
        wandb.log({"task_vector_histogram_plot": wandb.Image(fig)})

        wandb.finish()


if __name__ == "__main__":
    main()
