# This code is adapted from:
# https://github.com/mlfoundations/task_vectors
# Accessed on: 2024-03-09

import copy

import torch


class TaskVector:
    def __init__(
        self,
        pretrained_model=None,
        finetuned_model=None,
        vector=None,
        device_map="auto",
    ):
        self.device_map = device_map
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_model is not None and finetuned_model is not None
            with torch.no_grad():
                pretrained_state_dict = pretrained_model.state_dict()
                finetuned_state_dict = finetuned_model.state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_model, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            return_model = copy.deepcopy(pretrained_model)
            new_state_dict = {}
            pretrained_state_dict = return_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        return_model.load_state_dict(new_state_dict, strict=True)
        return return_model
