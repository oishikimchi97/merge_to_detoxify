import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM

from mtd.task_vector.negate import generate_negated_model
from mtd.utils.file import get_negated_dir, get_pretrained_path

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
        "--output_dir", type=str, default=None, help="Path to output file"
    )
    parser.add_argument("--coef", type=float, default=1.0)

    args = parser.parse_args()
    finetuned_path = Path(args.finetune_path)
    if args.pretrained_path:
        pretrained_path = args.pretrained_path
    else:
        pretrained_path = get_pretrained_path(finetuned_path)

    print(f"Using pretrained model from {pretrained_path}")

    coef = args.coef
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_negated_dir(finetuned_path, coef)

    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_path)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_path)

    negated_model = generate_negated_model(
        pretrained_model, finetuned_model, pretrained_path, finetuned_path, coef
    )
    negated_model.save_pretrained(output_dir)
    print(f"Negated model saved to {output_dir}")
