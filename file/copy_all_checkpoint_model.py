import argparse
import re
import shutil
from pathlib import Path

import tqdm

from utils import load_train_config


def get_step_num_from_path(x):
    return int(str(x).split("-")[-1])


def get_epoch_num(sub_dirs, checkpoint_idx):
    sub_dir = sub_dirs[0]
    checkpoint_dirs = [
        checkpoint_path
        for checkpoint_path in sub_dir.glob("*")
        if checkpoint_path.is_dir()
    ]
    sorted_checkpoint_path = sorted(checkpoint_dirs, key=get_step_num_from_path)
    checkpoint_dir_path = sorted_checkpoint_path[checkpoint_idx]
    train_config_path = checkpoint_dir_path / "trainer_state.json"
    train_config = load_train_config(train_config_path)
    epoch = int(train_config["num_train_epochs"])
    return epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub_dir", required=True, type=str, help="Path to sub directories"
    )
    parser.add_argument(
        "--dest_dir", default=None, type=str, help="Path to destination directory"
    )
    parser.add_argument(
        "--index",
        default=-1,
        type=int,
        help="""The checkpoint index you want to get the model from. The index starts from 1.
                The index -i denotes checkpoint[-i] and it will bring the last checkpoint model. The default value is set as 1""",
    )
    args = parser.parse_args()

    parent_dir = Path(args.sub_dir)
    sub_dirs = list(parent_dir.glob("*"))
    checkpoint_idx = args.index - 1 if args.index > 0 else args.index
    epoch = get_epoch_num(sub_dirs, checkpoint_idx)
    if args.dest_dir:
        dest_dir = Path(args.dest_dir)
    else:
        if "ep" in parent_dir.name:
            project_name = re.sub(r"ep(\d+)", f"ep{epoch}", parent_dir.name)
        else:
            project_name = f"{parent_dir.name}-ep{epoch}"

        dest_dir = parent_dir.parent / project_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for parent_dir in sub_dirs:
        checkpoint_dirs = [
            sub_dir for sub_dir in parent_dir.glob("*") if sub_dir.is_dir()
        ]
    # TODO: Add function to get models with the epoch user pass, not from the index.
    checkpoint_paths = [
        sorted(list(sub_dir.glob("*")), key=get_step_num_from_path)[checkpoint_idx]
        for sub_dir in sub_dirs
    ]

    for check_path in tqdm.tqdm(checkpoint_paths, desc="Copying checkpoint models"):
        dest_sub_dir = dest_dir / check_path.parent.name
        if not dest_sub_dir.exists():
            shutil.copytree(check_path, dest_sub_dir)
            print(f"Copied from {str(check_path)} to {str(dest_sub_dir)}")
        else:
            print(f"Directory {str(dest_sub_dir)} already exists. Skipping...")
    # For reading from shell script.
    print(str(dest_dir))
