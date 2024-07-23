import argparse

import wandb


def get_sweep_run_count(sweep_path):
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    return sweep.expected_run_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", help="Path to the sweep")
    args = parser.parse_args()

    sweep_path = args.sweep
    run_count = get_sweep_run_count(sweep_path)
    print(run_count)
