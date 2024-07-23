import argparse
from pathlib import Path

import numpy as np
import tqdm
from transformers import AutoModelForCausalLM

from mtd.task_vector.negate import generate_negated_model
from mtd.utils.file import get_negated_dir, get_pretrained_path

EPS = 1e-10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--finetune_path", required=True, type=str, help="Path to finetuned model"
    )
    parser.add_argument(
        "--min", type=float, default=0.0, help="Min value for coefficient"
    )
    parser.add_argument(
        "--max", type=float, default=1.0, help="Max value for coefficient"
    )
    parser.add_argument(
        "--step_size", type=float, default=0.05, help="Step size for coefficient"
    )

    args = parser.parse_args()
    finetuned_path = Path(args.finetune_path)
    if args.pretrained_path:
        pretrained_path = Path(args.pretrained_path)
    else:
        pretrained_path = get_pretrained_path(finetuned_path)

    print(f"Using pretrained model from {pretrained_path}")

    pretrained_model = AutoModelForCausalLM.from_pretrained(
        pretrained_path, device_map="auto"
    )
    finetune_model = AutoModelForCausalLM.from_pretrained(
        finetuned_path, device_map="auto"
    )

    coef_min = args.min
    coef_max = args.max
    # Create array from 0 to 1.0 with 0.05 stride
    coef_array = np.arange(coef_min, coef_max + EPS, args.step_size)[1:]

    for coef in tqdm.tqdm(coef_array, desc="generating negated models"):
        negated_model = generate_negated_model(
            pretrained_model, finetune_model, pretrained_path, finetuned_path, coef
        )
        negated_dir = get_negated_dir(finetuned_path, coef)
        negated_model.save_pretrained(negated_dir)
        print(f"Negated model saved to {negated_dir}")
