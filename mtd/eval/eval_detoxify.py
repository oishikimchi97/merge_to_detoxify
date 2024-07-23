import argparse
from typing import Dict, List, Optional

import pandas as pd
from detoxify import Detoxify

from mtd.eval.utils import compute_average_score, compute_ratio
from mtd.utils import PATH_TYPE

DETOXIFY_LABLES = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]


#!Caution: After Feb 15, the prompts will be excluded from the validation.
def eval_detoxify(
    continuation_path: PATH_TYPE,
    labels: Optional[List[str]] = None,
    include_prompt: bool = False,
) -> Dict[str, float]:
    if labels is None:
        labels = ["toxicity"]
    assert all(
        label in DETOXIFY_LABLES for label in labels
    ), f"Invalid label included: {labels}"
    df = pd.read_csv(continuation_path).fillna("")
    if include_prompt:
        sentences = (df["prompt"] + df["continuation"]).tolist()
    else:
        sentences = df["continuation"].tolist()
    # cleaned_sentences = delete_nan(sentences)

    result = Detoxify("unbiased", device="cuda").predict(sentences)
    score = {}
    for label in labels:
        score[label + " avg_score"] = compute_average_score(result[label])
        score[label + " ratio"] = compute_ratio(result[label])
    return score


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="Path to the CSV file")

    args = parser.parse_args()
    file_path = args.file_path

    avg_score, ratio_score = eval_detoxify(file_path)

    print("Average toxicity score: ", avg_score)
    print("Ratio of toxic continuations: ", ratio_score)
