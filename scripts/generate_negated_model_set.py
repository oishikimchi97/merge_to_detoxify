import argparse
from pathlib import Path

import tqdm
from transformers import AutoModelForCausalLM

from mtd.task_vector.negate import generate_negated_model
from mtd.utils.config import read_model_config
from mtd.utils.file import get_negated_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="Directory of the model")
    parser.add_argument(
        "--coef", default=1.0, type=float, help="Coefficient for negation"
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    coef = args.coef

    sub_model_dirs = [dir for dir in model_dir.glob("*") if dir.is_dir()]
    if len(sub_model_dirs) == 0:
        print(f"No sub-models found in {model_dir}")
    else:
        print(f"Found {len(sub_model_dirs)} sub-models in {model_dir}")

    pretrained_model = None
    for sub_model_dir in tqdm.tqdm(sub_model_dirs, desc="Negating models"):
        pretrained_model_name = read_model_config(sub_model_dir)["_name_or_path"]
        sub_model_output_dir = get_negated_dir(sub_model_dir, coef)

        if pretrained_model is None:
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, device_map="auto"
            )
        finetune_model = AutoModelForCausalLM.from_pretrained(
            sub_model_dir, device_map="auto"
        )

        negated_model = generate_negated_model(
            pretrained_model,
            finetune_model,
            pretrained_model_name,
            str(sub_model_dir),
            coef,
        )
        negated_model.save_pretrained(sub_model_output_dir)
        print(f"Negated model saved to {sub_model_output_dir}")
