"""
Training and validation lists of metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from ignite.metrics import Loss

from classifiers.metrics import metrics


class TrainMetrics:
    """
    Training metrics handling.

    Parameters
    ----------
    loss_fn : Callable
        Loss function.
    device : {"cpu", "cuda"}
        Device to run metrics on.

    Returns
    -------
    dict
        A dictionary of training metrics.
    """

    def __init__(self, loss_fn: Callable, device: Literal["cpu", "cuda"]):
        self.metrics = {
            "loss": Loss(loss_fn, device=device),
        }


class ValidMetrics:
    """
    Validation metrics handling.

    Parameters
    ----------
    loss_fn : Callable
        Loss function.
    device : {"cpu", "cuda"}
        Device to run metrics on.
    """

    def __init__(
        self,
        loss_fn: Callable,
        device: Literal["cpu", "cuda"],
    ) -> None:
        self.metrics = {
            "val_loss": Loss(loss_fn),
            # Torchmetrics "macro" accuracy (average of each label's accuracy)
            "val_acc": metrics.AccuracyMetric(device=device),
            # Torchmetrics accuracy for each label
            "f1_score": metrics.F1ScoreMetric(device=device),
            # Raw model output
            "output": metrics.ModelOutputMetric(device=device),
        }


class EvalMetrics:
    """
    Evaluation metrics handling.

    Parameters
    ----------
    device : {"cpu", "cuda"}
        Device to run metrics on.
    """

    def __init__(self, device: Literal["cpu", "cuda"]) -> None:
        self.metrics = {
            # Torchmetrics "macro" accuracy (average of each label's accuracy)
            "accuracy": metrics.AccuracyMetric(device=device),
            # Torchmetrics accuracy for each label
            "f1_score": metrics.F1ScoreMetric(device=device),
            # Torchmetrics confusion matrix
            "confusion_matrix": metrics.ConfusionMatrixMetric(device=device),
            # Torchemtrics precision for each label
            "precision": metrics.PrecisionMetric(device=device),
            # Torchmetrics recall for each label
            "recall": metrics.RecallMetric(device=device),
            # Torchmetrics precision-recall curve
            "precision_recall_curve": metrics.PrecisionRecallCurveMetric(device=device),
        }

    @staticmethod
    def summary(results: dict) -> dict:
        """
        Generate a summary of selected metrics from the results.

        Parameters
        ----------
        results : dict
            A dictionary containing all the evaluation metrics.

        Returns
        -------
        dict
            A dictionary containing only the selected summary metrics.

        Notes
        -----
        The summary includes the following metrics:
        - accuracy
        - f1_score
        """
        summary_results = ["accuracy", "f1_score"]
        return {k: v for k, v in results.items() if k in summary_results}
