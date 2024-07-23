from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_RANDOM_SEED = 42


def get_dataset_path(cfg: OmegaConf) -> Path:
    dataset_base_dir = Path(cfg.dataset.dataset_base_dir)
    split = cfg.dataset.split
    sub_dir_name = cfg.dataset.sub_dir_name
    return dataset_base_dir / split / sub_dir_name


def cal_one_epoch_step(
    train_dataset, per_device_batch_size, gradient_accumulation_steps=1, num_devices=1
):
    return len(train_dataset) // (
        per_device_batch_size * gradient_accumulation_steps * num_devices
    )


def get_train_run_name(cfg: OmegaConf):
    if cfg.dataset.sub_dir_name:
        run_name = f"{cfg.dataset.split}-{cfg.dataset.sub_dir_name}"
    else:
        run_name = cfg.dataset.split

    run_name = (
        run_name + f"-{cfg.seed}" if cfg.seed != DEFAULT_RANDOM_SEED else run_name
    )
    return run_name


def get_output_dir(model_name, project_name, run_name):
    if model_name in project_name:
        project_name = project_name.replace(model_name + "-", "")
    output_dir = f"results/{model_name}/{project_name}/{run_name}"
    return output_dir
