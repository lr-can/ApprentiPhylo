from abc import ABC
from pathlib import Path
from typing import Literal

import torch
from ignite.handlers import ProgressBar
from torch import Generator
from torch.utils.data import DataLoader, random_split

from classifiers.data.data import Data
from classifiers.logger import Logger
from classifiers.train import Training
from classifiers.utils import PADDING_TOKEN, RANDOM_SEED


def pad_collate_fn(batch: list) -> tuple:
    """
    Pad and collate a batch of sequences.

    Function used as custom collate function for pytorch dataloaders.

    Parameters
    ----------
    batch : list
        A list of tuples containing alignments and labels.

    Returns
    -------
    tuple
        A tuple containing padded alignments and labels as tensors.
    """

    def pad_align(align: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """
        Pad an alignment to the specified width and height.

        Parameters
        ----------
        align : torch.Tensor
            The alignment tensor to pad.
        width : int
            The target width.
        height : int
            The target height.

        Returns
        -------
        torch.Tensor
            The padded alignment tensor.
        """
        shape = align.shape
        new_shape = shape[0], width - shape[1]
        res = torch.cat([align, torch.full(new_shape, fill_value=PADDING_TOKEN, dtype=torch.int8)], dim=1)
        new_shape = height - shape[0], width
        res = torch.cat([res, torch.full(new_shape, fill_value=PADDING_TOKEN, dtype=torch.int8)], dim=0)
        return res

    batch.sort(key=lambda x: len(x[0]), reverse=True)
    aligns, labels = zip(*batch, strict=True)
    max_height = max([align.shape[0] for align in aligns])
    max_width = max([align.shape[1] for align in aligns])
    aligns = torch.stack([pad_align(align, max_width, max_height) for align in aligns])
    labels = torch.tensor(labels)
    return aligns, labels


class DeepClassifier(ABC):  # noqa: B024
    """
    Abstract base class for deep learning classifiers.

    Parameters
    ----------
    data : BaseData
        The data object containing the input dataset.
    out_path : str or Path
        The output path for saving results.
    device : {'cpu', 'cuda'}
        The device to use for computations.
    learning_rate : float
        The learning rate for the optimizer.
    batch_pad_sequences : bool, optional
        Whether to pad sequences in batches, by default False.
    progress_bar : bool, optional
        Whether to display a progress bar, by default True.
    disable_compile : bool, optional
        Whether to disable model compilation, by default True.
    split_proportion : float, optional
        The proportion of data to use for training, by default 0.9.
    batch_size : int, optional
        The batch size for training, by default 64.
    max_epochs : int, optional
        The maximum number of epochs for training, by default 200.
    early_stopping_patience : int, optional
        The number of epochs to wait before early stopping, by default 5.

    Raises
    ------
    TypeError
        If the data is not an instance of BaseData.
    RuntimeError
        If the data has not been preprocessed.
    """

    def __init__(
        self,
        data: Data,
        *,
        out_path: str | Path,
        device: Literal["cpu", "cuda"],
        learning_rate: float,
        batch_pad_sequences: bool = False,
        progress_bar: bool = True,
        disable_compile: bool = True,
        split_proportion: float = 0.9,
        batch_size: int = 64,
        max_epochs: int = 200,
        early_stopping_patience: int = 5,
    ) -> None:
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True)
        self.split_proportion = split_proportion
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.disable_compile = disable_compile

        if not isinstance(data, Data):
            msg = "Data must be an instance of BaseData."
            raise TypeError(msg)
        self.data = data

        self.model = None
        self.dataset = None

        self.progress_bar = ProgressBar() if progress_bar else None
        self.batch_pad_sequences = batch_pad_sequences
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience

        self.logger = Logger(
            log_path=self.out_path / "training.log",
            progress_bar=self.progress_bar,
            logger_name=f"{__name__}.train",
        )

    def get_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Create and return data loaders for training and validation.

        Returns
        -------
        tuple
            A tuple containing the training and validation data loaders.

        Raises
        ------
        ValueError
            If the dataset has not been defined.
        """
        if self.dataset is None:
            msg = "Dataset has not been defined"
            raise ValueError(msg)
        self.logger.log("--- Creating loaders ---")
        generator = Generator().manual_seed(RANDOM_SEED)
        train_dataset, valid_dataset = random_split(
            self.dataset, [self.split_proportion, 1 - self.split_proportion], generator=generator
        )
        collate_fn = pad_collate_fn if self.batch_pad_sequences else None
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn
        )
        return train_loader, valid_loader

    def train(self) -> None:
        """
        Train the model.

        Raises
        ------
        ValueError
            If the model has not been defined.
        """
        if self.model is None:
            msg = "Model has not been defined"
            raise ValueError(msg)

        self.logger.log("--- Hyperparameters ---")
        hyperparams = {
            "model": self.model,
            "split_proportion": self.split_proportion,
            "batch_pad_sequences": self.batch_pad_sequences,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
        }
        if hasattr(self, "kernel_size"):
            hyperparams["kernel_size"] = self.kernel_size
        self.logger.log_dict(hyperparams)

        training = Training(
            model=self.model,
            loaders=self.get_loaders(),
            out_path=self.out_path,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            learning_rate=self.learning_rate,
            progress_bar=self.progress_bar,
            logger=self.logger,
            disable_compile=self.disable_compile,
            device=self.device,  # type: ignore
        )
        training.train()
