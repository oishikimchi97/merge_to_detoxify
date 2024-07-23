import os
import re
from pathlib import Path

import hydra
import pandas as pd
import wandb
from omegaconf import OmegaConf
from transformers import set_seed

from mtd.eval.eval_detoxify import eval_detoxify
from mtd.eval.eval_perplexity import validate_perplexity
from mtd.eval.eval_toxigen import eval_toxigen_hatebert
from mtd.eval.generate_toxic_sentences import generate_toxic_continuation
from mtd.eval.utils import get_eval_project_name


def get_coef(dir_name):
    match = re.search(r"negated-(\d+\.\d+)", dir_name)
    if match:
        return float(match.group(1))
    return None


@hydra.main(version_base=None, config_path="../config", config_name="eval")
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    print(f"Random seed: {cfg.seed}")

    model_path = Path(cfg.model_path)
    os.environ["WANDB_DISABLED"] = "false" if cfg.wandb else "true"

    eval_cfg = cfg.eval

    eval_result = {}
    if "toxicity" in cfg.eval:
        toxicity_cfg = eval_cfg.toxicity
        toxic_prompt_cfg = toxicity_cfg.prompt
        continuation_path = generate_toxic_continuation(
            model_path,
            eval_cfg.toxicity.dataset_path,
            num_sample=1,
            prompt_label=toxic_prompt_cfg.prompt_label,
            prompt_toxic_label=toxic_prompt_cfg.prompt_toxic_label,
            prompt_text_field=toxic_prompt_cfg.prompt_text_field,
        )

        print(f"Made continuation for {model_path}")
        print(f"Validate toxicity for {continuation_path}")

        if eval_cfg.toxicity.name == "detoxify":
            toxicity_result = eval_detoxify(
                continuation_path,
            )
        elif eval_cfg.toxicity.name == "toxigen_hatebert":
            toxicity_result = eval_toxigen_hatebert(continuation_path)

        print(f"Total Score: \n{toxicity_result}")
        eval_result.update(toxicity_result)

    if "perplexity" in cfg.eval:
        print(f"Validate perplexity for {model_path}")
        perplexity = validate_perplexity(model_path, **eval_cfg.perplexity.args)
        print(f"Perplexity: {perplexity}")
        eval_result["perplexity"] = perplexity

    if cfg.wandb:
        if cfg.project_name != "":
            project_name = cfg.project_name
            run_name = "-".join(str(model_path).split("/")[1:])
        else:
            if "toxicity" in cfg.eval:
                project_name = get_eval_project_name(
                    model_path, toxicity_cfg.name, toxic_prompt_cfg.name
                )
            else:
                project_name = "eval-perplexity"

            run_name = model_path.name
        run = wandb.init(project=project_name)
        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

        coef = get_coef(run_name)
        if coef is not None:
            run.config.update({"coef": coef})

        run.name = run_name

        if "toxicity" in cfg.eval and cfg.log_text:
            df = pd.read_csv(continuation_path)
            prompt_result = wandb.Table(dataframe=df)
            wandb.log({"prompt_result": prompt_result})

        wandb.log(eval_result)
        wandb.log({"coefficient": coef})
        wandb.finish()


if __name__ == "__main__":
    main()
