import torch

from transformers import Trainer, TrainerCallback
from transformers import PreTrainedTokenizer


class TrainerWithGenToEval(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] = None,
    ):
        with torch.no_grad(), self.compute_loss_context_manager():
            inputs = self._prepare_inputs(inputs)
            if prediction_loss_only:  # False by default
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            sample_idxs = inputs.pop("sample_idxs")
            evaluation_kwargs = inputs.pop("evaluation_kwargs")
            evaluator = evaluation_kwargs.pop("evaluator")

            output_ids, det, meta = getattr(model, evaluator)(
                **inputs,
                **evaluation_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            return (None, output_ids.reshape(1, -1), sample_idxs)


class StopTrainingAfterOneStepCallback(TrainerCallback):
    """A callback that stops the training loop after one step for debugging purposes."""

    def on_step_end(self, args, state, control, **kwargs):
        """
        This method is called at the end of each step.
        """
        # Stop training after the first step
        control.should_training_stop = True
        return control


class StopEvaluationAfterOneStepCallback(TrainerCallback):
    """A callback that stops the evaluation loop after one step for debugging purposes."""

    def on_evaluate(self, args, state, control, **kwargs):
        """
        This method is called at the end of each evaluation step.
        """
        # Stop evaluation after the first step
        control.should_evaluate = False
        control.should_training_stop = True  # Ensures stopping if used during training
        return control
