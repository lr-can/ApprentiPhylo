"""
Metrics definitions.
"""

from collections.abc import Callable
from typing import Any, Literal

import polars as pl
import torch
from ignite.metrics import Metric as IgniteMetric
from torchmetrics import classification


class ModelOutputMetric(IgniteMetric):
    """
    A metric class that collects model outputs and targets for later analysis.

    Parameters
    ----------
    device : {'cpu', 'cuda'}
        The device to run the metric on.
    output_transform : callable, optional
        A function that transforms the output. Default is the identity function.

    Attributes
    ----------
    pred : list
        List to store predictions.
    target : list
        List to store targets.
    indices : list
        List to store sample indices for mapping back to filenames.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        output_transform: Callable = lambda x: x,
    ):
        super().__init__(output_transform=output_transform, device=device)
        self.pred = []
        self.target = []
        self.indices = []

    def reset(self):
        """
        Reset the metric state by clearing prediction and target lists.
        """
        self.pred = []
        self.target = []
        self.indices = []

    def update(self, output) -> None:
        """
        Update the metric state by appending new predictions and targets.

        Parameters
        ----------
        output : tuple
            A tuple containing predictions and targets.
            If a third element (indices) is provided, it will be stored.
        """
        pred, target = output[0].detach(), output[1].detach()
        self.pred.append(pred)
        self.target.append(target)
        # Try to capture indices if available (for filename mapping)
        if len(output) > 2:
            self.indices.append(output[2])

    def compute(self) -> pl.DataFrame:
        """
        Compute the final metric result.

        Returns
        -------
        pl.DataFrame
            A DataFrame containing logits, probabilities, predictions, and targets.
            If indices were captured, they will be included as 'index' column.

        Notes
        -----
        This method stacks all collected predictions and targets, applies sigmoid
        to get probabilities, and creates binary predictions based on a 0.5 threshold.
        """
        logits = torch.vstack(self.pred).squeeze()
        probs = torch.nn.Sigmoid()(logits)
        preds = (probs > 0.5).to(torch.int8)
        target = torch.vstack(self.target).squeeze().to(torch.int8)

        result_dict = {
            "logit": logits.cpu().numpy(),
            "prob": probs.cpu().numpy(),
            "pred": preds.cpu().numpy(),
            "target": target.cpu().numpy(),
        }
        
        # Add indices if available
        if self.indices:
            # Flatten indices if they are tensors
            indices_flat = []
            for idx in self.indices:
                if isinstance(idx, torch.Tensor):
                    indices_flat.extend(idx.cpu().tolist())
                else:
                    indices_flat.extend(idx if isinstance(idx, list) else [idx])
            result_dict["index"] = indices_flat[:len(logits)]

        return pl.DataFrame(result_dict)


class CustomMetric(IgniteMetric):
    """
    A custom metric class that wraps TorchMetric for use with Ignite.

    Parameters
    ----------
    metric : type[TorchMetric]
        The TorchMetric class to be wrapped.
    device : {"cpu", "cuda"}
        The device to run the metric on.
    output_transform : Callable, optional
        A function that transforms the output. Default is the identity function.
    *args
        Additional positional arguments passed to the TorchMetric constructor.
    **kwargs
        Additional keyword arguments passed to the TorchMetric constructor.
    """

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        output_transform: Callable = lambda x: x,
    ) -> None:
        super().__init__(output_transform=output_transform, device=device)

    def reset(self) -> None:
        """
        Reset the metric state.
        """
        self.metric.reset()
        super().reset()

    def update(self, output) -> None:
        """
        Update the metric state.

        Parameters
        ----------
        output : tuple
            A tuple containing predictions and targets.
        """

        pred, target = output[0].detach(), output[1].detach()
        self.metric.update(pred, target)

    def compute(self) -> Any:
        """
        Compute the metric.

        Returns
        -------
        Any
            The computed metric value.
        """

        res = self.metric.compute()
        return res


class AccuracyMetric(CustomMetric):
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.metric = classification.BinaryAccuracy().to(device)
        super().__init__(device=device)

    def compute(self):
        # Output as float instead of tensor to be JSON-serializable
        res = super().compute()
        return float(res)


class F1ScoreMetric(CustomMetric):
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.metric = classification.BinaryF1Score().to(device)
        super().__init__(device=device)

    def compute(self):
        # Output as float instead of tensor to be JSON-serializable
        res = super().compute()
        return float(res)


class ConfusionMatrixMetric(CustomMetric):
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.metric = classification.BinaryConfusionMatrix().to(device)
        super().__init__(device=device)


class PrecisionMetric(CustomMetric):
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.metric = classification.BinaryPrecision().to(device)
        super().__init__(device=device)


class RecallMetric(CustomMetric):
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.metric = classification.BinaryRecall().to(device)
        super().__init__(device=device)


class PrecisionRecallCurveMetric(CustomMetric):
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.metric = classification.BinaryPrecisionRecallCurve().to(device)
        super().__init__(device=device)