"""
Logistic regression classifier.
"""

import json
import time
from datetime import timedelta
from pathlib import Path

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from classifiers.data.data import Data, MsaCompositionData
from classifiers.logger import Logger


class LogisticRegressionClassifier:
    """
    A logistic regression classifier for multiple sequence alignment (MSA) composition data.

    Parameters
    ----------
    data : BaseData
        The input data, which must be an instance of BaseData and preprocessed.
    out_path : str or Path
        The path where output files will be saved.
    cv : int, optional
        The number of cross-validation folds (default is 5).
    scale_features : bool, optional
        Whether to scale features using StandardScaler (default is True).
    shuffle_data : bool, optional
        Whether to shuffle the data before training (default is False).

    Raises
    ------
    TypeError
        If the input data is not an instance of BaseData.
    RuntimeError
        If the input data has not been preprocessed.
    """

    def __init__(
        self,
        data: Data,
        out_path: str | Path,
        cv: int = 5,
        *,
        scale_features: bool = True,
        shuffle_data: bool = False,
    ) -> None:
        if not isinstance(data, Data):
            msg = "Data must be an instance of BaseData."
            raise TypeError(msg)

        self.cv = cv
        self.scale_features = scale_features
        self.shuffle_data = shuffle_data
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True)

        self.logger = Logger(
            log_path=self.out_path / "training.log",
            progress_bar=None,
            logger_name=f"{__name__}.train",
        )

        self.logger.log("--- Preprocessing data ---")
        self.data = MsaCompositionData(data)
        align_ids = self.data.aligns.keys()
        self.x = pl.DataFrame([self.data.aligns[align_id] for align_id in align_ids], orient="row")
        self.y = [self.data.labels[align_id] for align_id in align_ids]

    def train(self) -> dict:
        """
        Train the logistic regression classifier using cross-validation.

        Returns
        -------
        dict
            The cross-validation results containing scores and fitted estimators.
        """

        self.logger.log("--- Start training ---")
        self.logger.log(f"Number of cross validation folds: {self.cv}")
        self.logger.log(f"Shuffle data: {self.shuffle_data}")
        self.logger.log(f"Scale features: {self.scale_features}")

        start_time = time.time()

        x, y = self.x, self.y
        if self.shuffle_data:
            x, y = shuffle(x, y)  # type: ignore
        if self.scale_features:
            model = make_pipeline(StandardScaler(), LogisticRegression())
        else:
            model = make_pipeline(LogisticRegression())
        cv_result = cross_validate(
            model,
            x,  # type:ignore
            y,  # type:ignore
            cv=self.cv,
            return_estimator=True,
            scoring=["f1", "accuracy"],
        )

        # Training time in seconds
        self.training_time = round(time.time() - start_time)

        f1s = tuple(str(round(score, 4)) for score in cv_result["test_f1"])
        accuracies = tuple(str(round(score, 4)) for score in cv_result["test_accuracy"])
        self.logger.log(
            "--- Training ended ---\n"
            f"Training time: {timedelta(seconds=self.training_time)}\n"
            f"Fold F1 scores: {f1s}\n"
            f"Fold accuracies: {accuracies}\n"
        )

        # Save summary
        summary = {"training_time": self.training_time, "fold_f1_scores": f1s, "fold_accuracies": accuracies}
        (self.out_path / "summary.json").write_text(json.dumps(summary))

        # Export preds from first fold model
        preds_path = self.out_path / "best_preds.parquet"
        model = cv_result["estimator"][0]
        preds = pl.DataFrame(
            {"prob": model.predict_proba(self.x)[:, 1], "pred": model.predict(self.x), "target": self.y}
        )
        preds.write_parquet(preds_path)

        return cv_result
