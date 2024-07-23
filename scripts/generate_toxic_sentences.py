import os
from pathlib import Path

import hydra
import pandas as pd
import wandb
from omegaconf import OmegaConf
from transformers import set_seed

from mtd.eval.generate_toxic_sentences import generate_toxic_continuation


# You have to define the all model path you want to make generations for in the config file.
@hydra.main(
    version_base=None,
    config_path="config",
    config_name="generate_toxic_sentences",
)
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    print(f"Random seed: {cfg.seed}")

    model_path = Path(cfg.model_path)
    os.environ["WANDB_DISABLED"] = "false" if cfg.wandb else "true"

    toxicity_cfg = cfg.eval.toxicity
    toxic_prompt_cfg = toxicity_cfg.prompt

    num_sample = cfg.num_sample

    continuation_path = generate_toxic_continuation(
        model_path,
        toxicity_cfg.dataset_path,
        batch_size=cfg.batch_size,
        num_sample=num_sample,
        prompt_label=toxic_prompt_cfg.prompt_label,
        prompt_toxic_label=toxic_prompt_cfg.prompt_toxic_label,
        prompt_text_field=toxic_prompt_cfg.prompt_text_field,
    )

    if cfg.wandb:
        run = wandb.init(project="generation_toxic_sentences_with_realtoxicprompts")

        run_name = "-".join(cfg.model_path.split("/")[1:])
        run.name = run_name
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

        if cfg.log_text:
            df = pd.read_csv(continuation_path)
            result_table = wandb.Table(dataframe=df)
            wandb.log({"result": result_table})


if __name__ == "__main__":
    main()
