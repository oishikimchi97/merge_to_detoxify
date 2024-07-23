import torch
import tqdm

from mtd.task_vector.task_vector import TaskVector


def cal_task_vector_norm(pretrained_model, finetuned_model, p: int = 1):
    task_vector_state_dict = TaskVector(pretrained_model, finetuned_model).vector
    task_vector_norm = torch.norm(
        torch.cat(
            [
                para.flatten()
                for para in tqdm.tqdm(
                    task_vector_state_dict.values(),
                    desc=f"Calculating the task vector norm with p={p}",
                )
            ]
        ),
        p=p,
    )
    return task_vector_norm.cpu().item()
