from mtd.task_vector.task_vector import TaskVector


def generate_negated_model(
    pretrained_model,
    finetuned_model,
    pretrained_path,
    finetuned_path="",
    scaling_coef="",
    device_map="auto",
):
    """
    Generates a negated model based on the provided pretrained and finetuned models.

    Args:
        pretrained_model (Model): The pretrained model.
        finetuned_model (Model): The finetuned model.
        pretrained_path (str): The path to the pretrained model.
        finetuned_path (str, optional): The path to the finetuned model. Defaults to "".
        scaling_coef (str, optional): The scaling coefficient. Defaults to "".
        device_map (str, optional): The device mapping. Defaults to "auto".

    Returns:
        Model: The negated model.
    """
    task_vector = TaskVector(
        pretrained_model=pretrained_model,
        finetuned_model=finetuned_model,
        device_map=device_map,
    )

    negated_task_vector = -task_vector
    negated_model = negated_task_vector.apply_to(pretrained_model, scaling_coef)
    if pretrained_path and finetuned_path:
        negated_model.config.negated_info = {
            "pretrained_path": str(pretrained_path),
            "finetune_path": str(finetuned_path),
            "scaling_coef": scaling_coef,
        }
    return negated_model
