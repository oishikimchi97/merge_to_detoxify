from torch import nn
from trl import SFTTrainer


class CustomTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop("loss_type", "grad_descent")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "grad_descent":
            return super().compute_loss(model, inputs, return_outputs)

        elif self.loss_type == "grad_ascent":
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
            )

            # For Gradient Ascent, multiply loss by -1
            loss *= -1.0
            return (loss, outputs) if return_outputs else loss
