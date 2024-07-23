import argparse
from pathlib import Path

import tqdm
from transformers import AutoModelForCausalLM

from mtd.merge.average import average_model
from mtd.utils.config import read_model_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", help="Path to model directory to merge models from"
    )
    parser.add_argument("--output", default=None, help="Path to output model")
    parser.add_argument("--map_auto", default=True, help="Use auto map for device_map")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if args.output is None:
        output_dir = model_dir / "average"
    model_dirs = [dir for dir in model_dir.glob("*") if dir.is_dir()]
    print(f"Merging model from {[dir_path.name for dir_path in model_dirs]}")
    models = []
    for model_dir in tqdm.tqdm(model_dirs, desc="Loading models"):
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        models.append(model)
    print(f"Loaded {len(models)} models")

    merged_model = average_model(models)
    config = read_model_config(model_dirs[0])
    merged_model.config._name_or_path = config["_name_or_path"]

    merged_model.save_pretrained(output_dir)
    print(f"Saved to {str(output_dir)}")
