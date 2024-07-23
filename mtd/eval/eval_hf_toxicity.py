import argparse

import evaluate
import pandas as pd

from mtd.eval.utils import compute_average_score, compute_ratio
from mtd.utils import PATH_TYPE


def eval_toxicity(continuation_path: PATH_TYPE):
    df = pd.read_csv(continuation_path)
    sentences = (df["prompt"] + df["continuation"]).tolist()

    toxicity = evaluate.load("toxicity")

    result = toxicity.compute(predictions=sentences)

    avg_score = compute_average_score(result["toxicity"])
    ratio_score = compute_ratio(result["toxicity"])

    print("Average toxicity score: ", avg_score)
    print("Ratio of toxic continuations: ", ratio_score)

    return avg_score, ratio_score


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the CSV file")

    args = parser.parse_args()
    file_path = args.file_path

    eval_toxicity(file_path)
