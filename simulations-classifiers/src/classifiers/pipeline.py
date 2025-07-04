"""
Pipeline handling.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import torch

from classifiers import classif
from classifiers.classif.deep_classifier import DeepClassifier
from classifiers.data import tokenizers
from classifiers.data.data import Data
from classifiers.data.sources import FastaSource
from classifiers.data.tokenizers import Tokenizer
from classifiers.logger import default_logger, default_logger_formatter
from classifiers.utils import CLASSIFIERS


class Pipeline:
    """
    A class to handle the pipeline for classification tasks.

    Parameters
    ----------
    real_path : Path or str
        Path to the real data.
    sim_path : Path or str
        Path to the simulated data.
    tokenizer : Tokenizer
        Tokenizer to use for processing the data.
    out_path : str or Path
        Output path for storing results.
    progress_bar : bool, optional
        Whether to display progress bars, by default False.
    disable_compile : bool, optional
        Whether to disable torch.compile, by default False.
    """

    def __init__(
        self,
        *,
        real_path: Path | str,
        sim_path: Path | str,
        tokenizer: Tokenizer,
        out_path: str | Path,
        progress_bar: bool = False,
        disable_compile: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True)
        self.disable_compile = disable_compile

        self.base_data = Data(
            source_real=FastaSource(real_path),
            source_simulated=FastaSource(sim_path),
            tokenizer=self.tokenizer,
        )

        self.init_time = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005
        for handler in default_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                default_logger.removeHandler(handler)
        fh = logging.FileHandler(self.out_path / f"pipeline_{self.init_time}.log")
        fh.setFormatter(default_logger_formatter)
        default_logger.addHandler(fh)
        self.logger = default_logger
        self.progress_bar = progress_bar

        self.classifiers = []

    @staticmethod
    def from_config(
        config_path: str | Path, *, progress_bar: bool = False, disable_compile: bool = False
    ) -> Pipeline:
        """
        Create a Pipeline instance from a configuration file.

        Parameters
        ----------
        config_path : str or Path
            Path to the configuration file.
        progress_bar : bool, optional
            Whether to display progress bars, by default False.
        disable_compile : bool, optional
            Whether to disable torch.compile, by default False.

        Returns
        -------
        Pipeline
            An instance of the Pipeline class.
        """
        config_path = Path(config_path)
        cfg = json.loads(config_path.read_text())
        pipeline = Pipeline(
            real_path=cfg["real_path"],
            sim_path=cfg["sim_path"],
            out_path=cfg["out_path"],
            tokenizer=getattr(tokenizers, cfg["tokenizer"]),
            progress_bar=progress_bar,
            disable_compile=disable_compile,
        )

        shutil.copy(config_path, Path(cfg["out_path"]) / f"config_{pipeline.init_time}.json")

        for c in cfg["classifiers"]:
            args = c["args"] if "args" in c else {}
            out_dir = c["out_dir"] if "out_dir" in c else c["classifier"]
            pipeline.add_classifier(c["classifier"], out_dir=out_dir, **args)
        return pipeline

    def add_classifier(self, classifier: str, out_dir: str | None = None, *args, **kwargs) -> None:
        """
        Add a classifier to the pipeline.

        Parameters
        ----------
        classifier : str
            Name of the classifier to add (e.g. "DenseMsaClassifier")
        out_dir : str | None
            Output directory for the classifier, defaults to None. If not provided, the classifier
            name is used as output directory.
        *args : tuple
            Additional positional arguments for the classifier.
        **kwargs : dict
            Additional keyword arguments for the classifier.

        Raises
        ------
        ValueError
            If the specified classifier is not in the list of known classifiers.
        """
        if classifier not in CLASSIFIERS:
            msg = f"Unknown classifier :{classifier}."
            raise ValueError(msg)

        classifier_class = getattr(classif, classifier)
        is_deep = issubclass(classifier_class, DeepClassifier)
        classifier_fn = partial(classifier_class, *args, **kwargs)

        out_dir = out_dir or classifier
        out_path = self.out_path / out_dir
        if out_path.exists():
            self.logger.info(f"{out_dir} directory already exists, skipping.")
            return None

        self.classifiers.append(
            {
                "classifier": classifier,
                "classifier_fn": classifier_fn,
                "out_path": out_path,
                "is_deep": is_deep,
            }
        )

    def preprocess_data(self) -> None:
        """
        Preprocess the base data.
        """
        self.logger.info("--- Preprocessing base data ---")
        self.base_data.preprocess()

    def run(self) -> None:
        """
        Run the pipeline for all added classifiers.

        This method checks if the base data has been preprocessed, and if so,
        it runs each classifier in the pipeline. For each classifier, it sets up
        the necessary arguments, instantiates the classifier, and calls its train method.
        """
        for classifier in self.classifiers:
            self.logger.info(f"--- Start running {classifier["classifier"]} ---")
            classifier_args = {"data": self.base_data, "out_path": classifier["out_path"]}
            if classifier["is_deep"]:
                classifier_args.update({"device": self.device, "progress_bar": self.progress_bar})
            classifier_fn = classifier["classifier_fn"](**classifier_args)
            classifier_fn.train()
            self.logger.info(f"--- End running {classifier["classifier"]} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pipeline runner")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--no-progress", action="store_true", required=False, help="Disable progress bar")
    parser.add_argument("--no-compile", action="store_true", required=False, help="Disable torch.compile")
    args = parser.parse_args()

    pipeline = Pipeline.from_config(
        args.config, progress_bar=not args.no_progress, disable_compile=not args.no_compile
    )
    pipeline.preprocess_data()
    pipeline.run()
