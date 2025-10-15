"""
Torch model training.
"""

import copy
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import (
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
    create_lr_scheduler_with_warmup,
)
from torch import nn
from torch.utils.data import DataLoader

from classifiers.logger import Logger
from classifiers.metrics.lists import TrainMetrics, ValidMetrics


class Training:
    """
    Class for training a neural network model.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    loaders : tuple of DataLoader
        A tuple containing train and validation data loaders.
    out_path : Path or str
        The output path for saving training results.
    progress_bar : ProgressBar or None, optional
        Progress bar for displaying training progress.
    disable_compile : bool, optional
        Whether to disable model compilation.
    max_epochs : int
        Maximum number of training epochs.
    early_stopping_patience : int
        Number of epochs with no improvement after which training will be stopped.
    learning_rate : float
        Initial learning rate for the optimizer.
    logger : Logger
        Logger object for logging training information.
    device : {'cpu', 'cuda'}
        Device to use for training.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        loaders: tuple[DataLoader, DataLoader],
        out_path: Path | str,
        progress_bar: ProgressBar | None = None,
        disable_compile: bool = False,
        max_epochs: int,
        early_stopping_patience: int,
        learning_rate: float,
        logger: Logger,
        device: Literal["cpu", "cuda"],
    ) -> None:
        self._device = device
        self._disable_compile = disable_compile
        self.out_path = Path(out_path)

        self.model = model
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate

        self._progress_bar = progress_bar
        self.logger = logger

        self.best_epoch = {"epoch": None, "val_loss": float("inf")}
        self.training_history = []

        self.train_loader, self.valid_loader = loaders

    def train(self) -> None:
        """
        Execute the main training loop.

        This method sets up the model, loss function, optimizer, and scheduler,
        and then runs the training process using Ignite engines. It handles the
        training loop, evaluation, checkpointing, and logging of results.
        """

        torch.set_float32_matmul_precision("high")
        model = self.model.to(self._device)
        if not self._disable_compile:
            model = torch.compile(model)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)  # type: ignore
        scheduler = create_lr_scheduler_with_warmup(
            torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99**epoch),
            warmup_start_value=0,
            warmup_duration=5,
            warmup_end_value=self.learning_rate,
        )

        train_metrics = TrainMetrics(loss_fn=loss_fn, device=self._device)  # type: ignore
        val_metrics = ValidMetrics(loss_fn=loss_fn, device=self._device)  # type: ignore

        ## Ignite engines

        trainer = create_supervised_trainer(model, optimizer, loss_fn, self._device)  # type: ignore
        train_evaluator = create_supervised_evaluator(
            model,  # type: ignore
            metrics=train_metrics.metrics,  # type: ignore
            device=self._device,
        )
        val_evaluator = create_supervised_evaluator(model, metrics=val_metrics.metrics, device=self._device)  # type: ignore

        # Run scheduler at each epoch start
        trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)  # type: ignore
        # Attach progress bar
        if self._progress_bar is not None:
            self._progress_bar.attach(trainer)

        ## Checkpointing and early stopping handlers

        def score_function(engine):
            val_loss = engine.state.metrics["val_loss"]
            return -val_loss

        model_checkpoint_handler = ModelCheckpoint(
            self.out_path / "checkpoint",
            n_saved=1,
            filename_prefix="best",
            score_function=score_function,
            score_name="neg_val_loss",
        )

        val_evaluator.add_event_handler(
            Events.COMPLETED,
            model_checkpoint_handler,
            to_save={
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "loss_fn": loss_fn,
            },
        )

        early_stopping_handler = EarlyStopping(
            patience=self.early_stopping_patience, score_function=score_function, trainer=trainer
        )
        val_evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

        ## Metrics handlers

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(_):
            train_evaluator.run(self.train_loader)
            val_evaluator.run(self.valid_loader)

        @val_evaluator.on(Events.COMPLETED)
        def save_best_metrics(engine):
            current_val_metrics = copy.deepcopy(engine.state.metrics)
            current_val_loss = current_val_metrics["val_loss"]
            if current_val_loss < self.best_epoch["val_loss"]:
                self.best_epoch = {
                    "epoch": trainer.state.epoch,
                    "val_loss": current_val_loss,
                    "metrics": current_val_metrics,
                }

        ## Logging handlers

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch_results(engine):  # noqa: ARG001
            current_epoch = trainer.state.epoch
            current_train_metrics = train_evaluator.state.metrics
            current_val_metrics = val_evaluator.state.metrics
            values = {
                "epoch": current_epoch,
                "train_loss": current_train_metrics["loss"],
                "val_loss": current_val_metrics["val_loss"],
                "val_acc": current_val_metrics["val_acc"],
                "f1": current_val_metrics["f1_score"],
                "lr": optimizer.param_groups[0]["lr"],
                "best": "*" if current_epoch == self.best_epoch["epoch"] else " ",
            }
            # Log epoch values
            self.logger.log(
                f"{values['epoch']:5}"
                f"{values['train_loss']:13.3f}"
                f"{values['val_loss']:13.3f}"
                f"{values['val_acc']:12.3f}"
                f"{values['f1']:12.3f}"
                f"{values['lr']:10.5f}"
                f"{values['best']:>7}"
            )
            # Keep track of epoch values
            self.training_history.append(values)

        ## Training

        if self._device == "cuda":
            self.logger.log_nvidia_smi()
        self._n_params = sum(p.numel() for p in model.parameters())
        self.logger.log(
            "--- Training start ---\n"
            f"Start training using {self._device} device.\n"
            f"Number of model parameters: {self._n_params}.\n"
            "Epoch   Train loss   Valid loss   Valid acc    F1-score        Lr   Best\n"
            "------------------------------------------------------------------------"
        )
        start_time = time.time()

        trainer.run(self.train_loader, max_epochs=self.max_epochs)

        # Training time in seconds
        self.training_time = round(time.time() - start_time)
        best_epoch = self.best_epoch["epoch"]
        best_metrics = self.best_epoch["metrics"]
        self.logger.log(
            "--- Training ended ---\n"
            f"Number of model parameters: {self._n_params}\n"
            f"Training time: {timedelta(seconds=self.training_time)}\n"
            f"Best epoch: {best_epoch}\n"
            f"Best valid loss: {best_metrics['val_loss']:.3f}\n"
            f"Best valid accuracy (macro): {best_metrics['val_acc']:.3f}\n"
            f"Best F1 score: {best_metrics['f1_score']:.3f}"
        )
        self.save_summary()
        self.save_train_history()
        self.save_best_preds()

    def save_train_history(self):
        """
        Save the training history to a Parquet file.

        This method saves the complete training history, including metrics for
        each epoch, to a Parquet file for later analysis.
        """

        train_df = pl.DataFrame(self.training_history)
        train_df.write_parquet(self.out_path / "train_history.parquet")

    def save_summary(self) -> None:
        """
        Save a summary of the training results to a JSON file.

        This method creates a summary dictionary containing the best epoch
        information and training time, and saves it to a JSON file.
        """
        summary = {
            "n_params": self._n_params,
            "best_epoch": self.best_epoch["epoch"],
            "training_time": self.training_time,
        }

        best_metrics = self.best_epoch["metrics"]
        summary.update({k: v for k, v in best_metrics.items() if k != "output"})

        (self.out_path / "summary.json").write_text(json.dumps(summary))

    def save_best_preds(self) -> None:
        """
        Save predictions and targets for the best model to a parquet file.

        This method saves the predictions and targets from the best performing
        epoch to a Parquet file for further analysis or evaluation.
        """

        best_preds_path = self.out_path / "best_preds.parquet"
        self.best_epoch["metrics"]["output"].write_parquet(best_preds_path)
