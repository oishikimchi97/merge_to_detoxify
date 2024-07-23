import argparse
from pathlib import Path

import pandas as pd
import tqdm


def main(result_dir: str, output_dir: str):
    if isinstance(result_dir, str):
        result_dir = Path(result_dir)

    result_paths = list(result_dir.glob("**/*.csv"))
    print(f"Found {len(result_paths)} result files")

    for result_path in tqdm.tqdm(result_paths):
        result_df = pd.read_csv(result_path)
        group_df = result_df.groupby("prompt_id")
        sampled_df = group_df.first()

        path_parts = list(result_path.parts)
        path_parts[0] = output_dir
        output_path = Path("/".join(path_parts))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sampled_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_dir", type=Path, help="Directory containing result files"
    )
    parser.add_argument(
        "--output_dir", default="sampled_result", type=str, help="Output file path"
    )
    args = parser.parse_args()

    main(args.result_dir, args.output_dir)
