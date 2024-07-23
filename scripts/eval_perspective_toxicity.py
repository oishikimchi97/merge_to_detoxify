import os
from pathlib import Path

import hydra
import pandas as pd
import wandb
from omegaconf import OmegaConf
from transformers import set_seed

from mtd.eval.eval_perspective import eval_perspective
from mtd.eval.generate_toxic_sentences import get_continuation_path


@hydra.main(
    version_base=None, config_path="../config", config_name="eval_perspective_toxicity"
)
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    print(f"Random seed: {cfg.seed}")

    os.environ["WANDB_DISABLED"] = "false" if cfg.wandb else "true"

    model_path = Path(cfg.model_path)
    dataset_path = Path(cfg.eval.toxicity.dataset_path)

    num_sample = cfg.num_sample

    continuation_path = get_continuation_path(model_path, dataset_path, num_sample)

    assert (
        continuation_path.exists()
    ), f"{continuation_path} does not exist. You have to make the continuation first using generate_toxic_sentences"

    expected_max_toxicity_score, probability_toxicity_score, result_path = (
        eval_perspective(
            continuation_path, model_path, num_sample, cfg.toxicity_threshold
        )
    )

    if cfg.wandb:
        run = wandb.init(project="eval-perspective")

        run_name = "-".join(cfg.model_path.split("/")[1:])
        run.name = run_name
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

        if cfg.log_text:
            df = pd.read_csv(result_path)
            result_table = wandb.Table(dataframe=df)
            wandb.log({"result": result_table})

        wandb.log(
            {
                "expected_max_toxicity_score": expected_max_toxicity_score,
                "probability_toxicity_score": probability_toxicity_score,
            }
        )


if __name__ == "__main__":
    main()
