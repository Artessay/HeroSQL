import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat
from typing import Literal

class BinaryBrierScore(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none", None]="mean",
        **kwargs
    ) -> None:
        """Compute the Brier score for binary classification.

        The Brier score is the mean squared error between probabilistic predictions
        and the binary outcomes (0 or 1). Lower values indicate better calibrated predictions.

        Args:
            reduction: How to aggregate across a batch:
                - 'mean': return mean Brier score (default)
                - 'sum': return summed Brier score
                - 'none' or None: return per-sample Brier scores
            kwargs: Additional torchmetrics Metric arguments.

        Inputs:
            - preds: probabilities for the positive class, shape (B,) or (B,1)
            - target: ground-truth binary labels, shape (B,)

        Example:
            >>> from your_metrics_package import BinaryBrierScore
            >>> metric = BinaryBrierScore(reduction="mean")
            >>> preds = torch.tensor([0.8, 0.3, 0.6, 0.1])
            >>> target = torch.tensor([1, 0, 1, 0])
            >>> metric.update(preds, target)
            >>> print(metric.compute())
            tensor(0.0950)
        """
        super().__init__(**kwargs)
        allowed_reductions = ("mean", "sum", "none", None)
        if reduction not in allowed_reductions:
            raise ValueError(f"reduction must be one of {allowed_reductions}, got {reduction}")
        self.reduction = reduction
        # State to accumulate results
        if reduction in ["mean", "sum"]:
            self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("scores", default=[], dist_reduce_fx="cat")

    def reset(self) -> None:
        """Reset the metric state."""
        if self.reduction in ["mean", "sum"]:
            self.score_sum = torch.tensor(0.0)
            self.num_samples = torch.tensor(0)
        else:
            self.scores = []

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the metric state with new batch predictions and targets.

        Args:
            preds: probabilities for class 1, shape (B,) or (B, 1)
            target: binary labels, shape (B,)
        """
        # Check input
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if preds.ndim != 1:
            raise ValueError(f"`preds` must be 1D tensor of probabilities, got shape {preds.shape}")
        if target.shape != preds.shape:
            raise ValueError(f"`target` must have same shape as `preds` ({preds.shape}), got {target.shape}")
        if not ((preds >= 0) & (preds <= 1)).all():
            raise ValueError("`preds` must be probabilities in [0, 1].")
        if not ((target == 0) | (target == 1)).all():
            raise ValueError("`target` must be binary: 0 or 1 only.")

        brier = (preds - target.float()) ** 2

        if self.reduction in [None, "none"]:
            self.scores.append(brier)
        else:
            self.score_sum += brier.sum()
            self.num_samples += brier.numel()

    def compute(self) -> Tensor:
        """Returns the final Brier score."""
        if self.reduction == "sum":
            return self.score_sum.clone()
        elif self.reduction == "mean":
            # Avoid division by zero
            if self.num_samples == 0:
                return torch.tensor(float("nan"))
            return self.score_sum.clone() / self.num_samples
        else:
            scores = dim_zero_cat(self.scores)
            return scores

# Example usage:
if __name__ == "__main__":
    metric = BinaryBrierScore(reduction="mean")
    preds = torch.tensor([0.8, 0.3, 0.6, 0.1])
    target = torch.tensor([1, 0, 1, 0])
    metric.update(preds, target)
    print("Brier score:", metric.compute())