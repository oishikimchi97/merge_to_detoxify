import argparse
from pathlib import Path

import pandas as pd

import wandb
from mtd.eval.eval_perspective import (
    cal_max_prob_toxicity_score,
    get_tox_score_per_prompt,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_path", help="Path to the dataframe")
    parser.add_argument(
        "--num_sample", default=25, help="Number of samples to be used", type=int
    )
    parser.add_argument(
        "--toxicity_threshold", default=0.5, help="Toxicity threshold", type=float
    )
    args = parser.parse_args()

    generations_score_df_path = Path(args.df_path)
    generations_score_df = pd.read_csv(generations_score_df_path)

    num_sample = args.num_sample
    toxicity_threshold = args.toxicity_threshold

    tox_score_per_prompt = get_tox_score_per_prompt(generations_score_df, num_sample)

    expected_max_toxicity_score, probability_toxicity_score = (
        cal_max_prob_toxicity_score(tox_score_per_prompt, toxicity_threshold)
    )

    run = wandb.init(project="eval-perspective")

    model_path = generations_score_df_path.parent.parent

    run_name = "-".join(str(model_path).split("/")[1:])
    run.name = run_name

    wandb.log(
        {
            "expected_max_toxicity_score": expected_max_toxicity_score,
            "probability_toxicity_score": probability_toxicity_score,
        }
    )
    print("Expected Max Toxicity Score: ", expected_max_toxicity_score)
    print("Probability Toxicity Score: ", probability_toxicity_score)

    wandb.finish()


if __name__ == "__main__":
    main()
