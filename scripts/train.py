import os

import datasets
import hydra
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from utils import (
    cal_one_epoch_step,
    get_dataset_path,
    get_output_dir,
    get_train_run_name,
)

from mtd.train.trainer import CustomTrainer


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: OmegaConf):
    set_seed(cfg.seed)
    print(f"Random seed: {cfg.seed}")

    project_name = cfg.project_name

    dataset_path = get_dataset_path(cfg)

    os.environ["WANDB_DISABLED"] = "false" if cfg.wandb else "true"
    local_rank = None

    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        print(f"Device map: {device_map}")
        group_name = get_train_run_name(cfg)
        run_name = f"{group_name}-RANK{local_rank}"
        output_dir = get_output_dir(cfg.model.name, project_name, group_name)
        if local_rank == 0:
            wandb.init(
                project=project_name,
                group=group_name,
                settings=wandb.Settings(start_method="thread"),
            )
    else:
        print("LOCAL_RANK is None")
        run_name = get_train_run_name(cfg)
        wandb.init(project=project_name)
        output_dir = get_output_dir(cfg.model.name, project_name, run_name)

    if local_rank == 0 or os.environ.get("LOCAL_RANK") is None:
        wandb.run.name = run_name

    model_id = cfg.model.hf_key
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_args = instantiate(cfg.model.args) if cfg.model.args is not None else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        **model_args,
    )

    model.generation_config.do_sample = True
    model.config.use_cache = False

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = datasets.load_from_disk(dataset_path)
    if "val" in cfg.dataset.hf_splits:
        train_dataset, val_dataset = dataset["train"], dataset["val"]
    else:
        # For those dataset that do not have validation set (e.g Toxigen).
        train_dataset = dataset["train"]
        val_dataset = None
        cfg.model.trainer.args.evaluation_strategy = "no"
        cfg.model.trainer.args.do_eval = False

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    trainer_cfg = cfg.model.trainer

    per_device_batch_size = trainer_cfg.args.per_device_train_batch_size
    gradient_accumulation_steps = trainer_cfg.args.gradient_accumulation_steps
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    print(f"num_devices: {num_devices}")

    one_epoch_step = cal_one_epoch_step(
        train_dataset, per_device_batch_size, gradient_accumulation_steps, num_devices
    )
    global_batch_size = per_device_batch_size * num_devices
    save_epoch = trainer_cfg.save_epoch
    save_steps = one_epoch_step * save_epoch
    max_seq_length = 128

    if local_rank == 0 or local_rank is None:
        wandb.config.update(
            {
                "one_epoch_step": one_epoch_step,
                "global_batch_size": global_batch_size,
                "num_devices": num_devices,
                "max_seq_length": max_seq_length,
            }
        )

    training_arguments = TrainingArguments(
        **trainer_cfg.args,
        output_dir=output_dir,
        save_strategy="no",
        # save_steps=save_steps,
        save_only_model=True,
        logging_steps=max(1, one_epoch_step // 20),
        report_to="wandb" if cfg.wandb else None,
    )

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field=cfg.dataset.text_field,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        loss_type=trainer_cfg.loss_type,
    )

    trainer.train()
    trainer.save_model(output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
