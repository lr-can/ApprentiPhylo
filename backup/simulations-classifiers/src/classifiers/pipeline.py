"""
Two-iteration classification pipeline (Run1 → filtering → Run2).

Stable version:
    • Compatible with your Data(), SequencesData, MsaCompositionData, SiteCompositionData
    • AACnnClassifier: custom prediction loop, no DeepClassifier.predict() call
    • LogisticRegressionClassifier: tries to train, skips cleanly if only 1 class (LR-2)
    • DenseSiteClassifier / DenseMsaClassifier: skipped for prediction
    • No import cycles
    • Per-classifier run_1 / run_2 folders preserved via out_dir
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import logging
import shutil
import argparse
import json

import torch
import polars as pl
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score

# ---------------------------------------------------------------------
# PYTHONPATH FIX
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------
from classifiers import classif
from classifiers.classif.deep_classifier import DeepClassifier
from classifiers.utils import CLASSIFIERS
from classifiers.logger import default_logger, default_logger_formatter

from classifiers.data.sources import FastaSource
from classifiers.data.data import Data
from classifiers.utils import LABEL_REAL, LABEL_SIMULATED

# =====================================================================
# CONFIG
# =====================================================================

# Classifiers which are NOT used for prediction on simulated alignments
NON_PREDICTABLE = {
    "DenseMsaClassifier",
    "DenseSiteClassifier",
}


# =====================================================================
#                           PIPELINE
# =====================================================================
class Pipeline:
    def __init__(
        self,
        *,
        real_path: str | Path,
        sim_path: str | Path,
        tokenizer,
        out_path: str | Path,
        progress_bar: bool = False,
        disable_compile: bool = False,
        threshold: float = 0.9,
    ):
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Common config
        self.tokenizer = tokenizer
        self.progress_bar = progress_bar
        self.disable_compile = disable_compile
        self.threshold = threshold

        # Output
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True)

        # Base dataset (real + simulated)
        self.base_data = Data(
            source_real=FastaSource(real_path),
            source_simulated=FastaSource(sim_path),
            tokenizer=self.tokenizer,
        )

        # Logger
        self.init_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._setup_logger()

        # Classifier registry
        self.classifiers: list[dict] = []

        self.logger.info(f"Pipeline initialized at {self.init_time}")
        self.logger.info(f"Device   : {self.device}")
        self.logger.info(f"Real data: {real_path}")
        self.logger.info(f"Sim data : {sim_path}")
        self.logger.info(f"Output   : {self.out_path}")

    # -----------------------------------------------------------------
    def _setup_logger(self) -> None:
        # Remove previous file handlers if any
        for h in list(default_logger.handlers):
            if isinstance(h, logging.FileHandler):
                default_logger.removeHandler(h)

        log_file = self.out_path / f"pipeline_{self.init_time}.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(default_logger_formatter)
        default_logger.addHandler(fh)

        self.logger = default_logger
    
    def _has_two_classes(self) -> bool:
        """Return True if current dataset contains at least two classes."""
        labels = list(self.base_data.labels.values())
        return len(set(labels)) >= 2


    # =================================================================
    #                       LOAD FROM CONFIG
    # =================================================================
    @staticmethod
    def from_config(
        config_path: str | Path,
        *,
        progress_bar: bool = False,
        disable_compile: bool = False,
        threshold: float = 0.5,
    ) -> "Pipeline":
        config_path = Path(config_path)
        cfg = json.loads(config_path.read_text())

        # Proper tokenizer import (class or instance, as in your code)
        from classifiers.data import tokenizers as tok_module

        tokenizer = getattr(tok_module, cfg["tokenizer"])

        pipeline = Pipeline(
            real_path=cfg["real_path"],
            sim_path=cfg["sim_path"],
            out_path=cfg["out_path"],
            tokenizer=tokenizer,
            progress_bar=progress_bar,
            disable_compile=disable_compile,
            threshold=threshold,
        )

        # Save config snapshot
        shutil.copy(config_path, Path(cfg["out_path"]) / f"config_{pipeline.init_time}.json")

        # Register classifiers from config
        for c in cfg["classifiers"]:
            args = c.get("args", {})
            out_dir = c.get("out_dir", c["classifier"])
            pipeline.add_classifier(c["classifier"], out_dir=out_dir, **args)

        return pipeline

    # =================================================================
    #                        REGISTER CLASSIFIER
    # =================================================================
    def add_classifier(self, name: str, out_dir: str | None = None, *args, **kwargs) -> None:
        """
        Register a classifier to the pipeline.

        Parameters
        ----------
        name : str
            Classifier name key from CLASSIFIERS.
        out_dir : str, optional
            Subdirectory name for this classifier under run_1 / run_2.
        """
        if name not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier: {name}")

        clf_class = getattr(classif, name)
        is_deep = issubclass(clf_class, DeepClassifier)

        # we keep **kwargs so we can reuse hyperparams for prediction
        def make_clf(**call_kwargs):
            merged = {**kwargs, **call_kwargs}
            return clf_class(**merged)

        self.classifiers.append(
            {
                "classifier": name,
                "classifier_fn": make_clf,
                "kwargs": kwargs,
                "out_dir": out_dir or name,
                "is_deep": is_deep,
            }
        )

    # =================================================================
    #                           TRAINING
    # =================================================================
    def train_classifier(self, clf: dict, out_dir: Path) -> None:
        clf_name = clf["classifier"]
        best_model = out_dir / "best_model.pt"

        # ----------------------------------------------------------
        # Skip si modèle déjà entraîné
        # ----------------------------------------------------------
        if best_model.exists():
            self.logger.info(f"[RUN] Skipping training {clf_name} (best_model exists)")
            return

        self.logger.info(f"[RUN] Training {clf_name}")

        # ----------------------------------------------------------
        # Instanciation du classifieur
        # ----------------------------------------------------------
        args = {
            "data": self.base_data,
            "out_path": out_dir,
            **clf["kwargs"],
        }

        if clf["is_deep"]:
            args["device"] = self.device
            args["progress_bar"] = self.progress_bar
            args["disable_compile"] = self.disable_compile

        model = clf["classifier_fn"](**args)

        # ==================================================================
        #     SKIP GLOBAUX si 1 seule classe (RUN2 par ex)
        # ==================================================================
        # Data doit fournir labels dans self.base_data.labels
        labels = getattr(self.base_data, "labels", None)
        if labels is not None:
            unique_classes = set(labels.values())

            if len(unique_classes) < 2:
                if clf["is_deep"]:
                    self.logger.info(
                        f"[RUN] Skipping {clf_name} (deep model — single class)"
                    )
                    return

                if clf_name == "LogisticRegressionClassifier":
                    self.logger.info(
                        f"[RUN] Skipping {clf_name} (logreg — single class)"
                    )
                    return

        # ==================================================================
        #     COLLATE PATCH (x, y) Universel pour Dense* et AACnn
        # ==================================================================
        if clf_name in ("DenseMsaClassifier", "DenseSiteClassifier", "AACnnClassifier"):

            def collate_x_y(batch):
                # batch = [(x,y), (x,y), ...] ou [(x, y, extra), ...]
                xs = torch.stack([b[0] for b in batch])
                ys = torch.tensor([b[1] for b in batch]).float().unsqueeze(1)
                return xs, ys

            def patched_get_loaders():
                from torch.utils.data import DataLoader
                train_loader = DataLoader(
                    model.dataset,
                    batch_size=model.batch_size,
                    shuffle=True,
                    collate_fn=collate_x_y,
                )
                valid_loader = DataLoader(
                    model.dataset,
                    batch_size=model.batch_size,
                    shuffle=False,
                    collate_fn=collate_x_y,
                )
                return train_loader, valid_loader

            model.get_loaders = patched_get_loaders
            self.logger.info(f"[RUN] Applied collate_x_y patch to {clf_name}")

        # ==================================================================
        #     TRAINING
        # ==================================================================
        if clf_name == "LogisticRegressionClassifier":
            try:
                model.train()
            except ValueError as e:
                self.logger.warning(
                    "[RUN] LogisticRegressionClassifier training skipped "
                    f"(likely single class). Error: {e}"
                )
                return
        else:
            model.train()

        self.logger.info(f"[RUN] Finished training {clf_name}")




    # =================================================================
    #                           PREDICTION
    # =================================================================
    def predict_on_sim(self, clf: dict, clf_dir: Path):
        """
        Predict on simulated alignments for a given classifier.
        """
        clf_name = clf["classifier"]
        model_path = clf_dir / "best_model.pt"

        if not model_path.exists():
            self.logger.warning(f"[RUN] Skipping {clf_name}: no best_model.pt")
            return None

        # Skip models that are not used for prediction on simulations
        if clf_name in NON_PREDICTABLE:
            self.logger.info(f"[RUN] Skipping {clf_name} (prediction not supported)")
            return None

        # RUN2: skip deep models if dataset has 1 class (except AACnn)
        if hasattr(self, "_current_iteration"):
            if self._current_iteration == 2 and not self._has_two_classes():
                if clf["is_deep"] and clf_name != "AACnnClassifier":
                    return None
                if clf_name == "LogisticRegressionClassifier":
                    return None

        if clf_name == "LogisticRegressionClassifier":
            return self._predict_logreg(clf, clf_dir, model_path)

        if clf_name == "AACnnClassifier":
            return self._predict_aacnn(clf, clf_dir, model_path)

        # by default: no prediction
        self.logger.info(f"[RUN] Skipping {clf_name} (no prediction implementation)")
        return None


    # --- Logistic Regression prediction --------------------------------
    def _predict_logreg(self, clf: dict, clf_dir: Path, model_path: Path):
        self.logger.info("[RUN] Predicting LogisticRegressionClassifier…")

        from classifiers.classif import LogisticRegressionClassifier

        # rebuild classifier
        clf_obj = LogisticRegressionClassifier(
            data=self.base_data,
            out_path=clf_dir,
            **clf["kwargs"],
        )

        # relies on its own predict() implementation
        return clf_obj.predict(model_path=model_path)

    # --- AACnn prediction (custom, no DeepClassifier.predict) ----------
    def _predict_aacnn(self, clf: dict, clf_dir: Path, model_path: Path):
        """
        Custom prediction loop for AACnnClassifier:
        - builds a Data object containing **both real and simulated alignments** (for ROC calculation)
        - instantiates AACnnClassifier on this data
        - loads best_model.pt
        - runs a manual forward pass, handling 1 or 2 output neurons
        - returns a polars.DataFrame(filename, prob_real, pred_class)
        """
        self.logger.info("[RUN] Predicting AACnnClassifier…")

        from classifiers.classif import AACnnClassifier
        from classifiers.data.sources import FastaSource
        from classifiers.data.data import Data
        from torch.utils.data import DataLoader
        import copy

        # --- Build Data with both real and simulated alignments (for ROC) ---
        real_src = FastaSource(self.base_data.source_real.data_path)
        sim_src = FastaSource(self.base_data.source_simulated.data_path)

        full_data = Data(
            source_real=real_src,
            source_simulated=sim_src,
            tokenizer=self.tokenizer,
        )

        # Copy kwargs (e.g. max_width, kernel_size, etc.)
        clf_kwargs = copy.deepcopy(clf["kwargs"])

        # Instantiate classifier (architecture + dataset only)
        clf_obj = AACnnClassifier(
            data=full_data,
            out_path=clf_dir,
            device=self.device,
            progress_bar=False,
            disable_compile=True,
            **clf_kwargs,
        )

        # --- Load model weights -----------------------------------------
        state = torch.load(model_path, map_location=self.device)
        clf_obj.model.load_state_dict(state)
        clf_obj.model.to(self.device)
        clf_obj.model.eval()

        # --- Build dataloader over the AACnn dataset --------------------
        dataset = clf_obj.dataset
        batch_size = clf_kwargs.get("batch_size", 64)

        # Collate: (align, label)
        def _collate(batch):
            aligns = torch.stack([b[0] for b in batch])
            labels = torch.stack([b[1] for b in batch])
            return aligns, labels

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

        # Filenames for mapping outputs
        filenames_all = dataset.keys
        idx = 0

        filenames = []
        prob_real_list = []
        pred_class_list = []

        # --- Prediction loop --------------------------------------------
        with torch.no_grad():
            for batch_aligns, batch_labels in loader:
                batch_aligns = batch_aligns.to(self.device)

                logits = clf_obj.model(batch_aligns)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(1)

                n_classes = logits.shape[1]

                if n_classes == 1:
                    prob_real_batch = torch.sigmoid(logits.squeeze(1))
                    pred_class_batch = (prob_real_batch >= 0.5).long()
                elif n_classes == 2:
                    probs = torch.softmax(logits, dim=1)
                    prob_real_batch = probs[:, 1]
                    pred_class_batch = probs.argmax(dim=1)
                else:
                    raise RuntimeError(
                        f"AACnnClassifier model has unsupported output dimension: {n_classes}"
                    )

                # Filenames batch
                batch_size_actual = batch_aligns.size(0)
                batch_names = filenames_all[idx: idx + batch_size_actual]
                idx += batch_size_actual

                # Append
                filenames.extend(batch_names)
                prob_real_list.extend(prob_real_batch.detach().cpu().numpy().tolist())
                pred_class_list.extend(pred_class_batch.detach().cpu().numpy().tolist())

        df = pl.DataFrame(
            {
                "filename": filenames,
                "prob_real": prob_real_list,
                "pred_class": pred_class_list,
            }
        )

        return df
    
    def _save_selected_predictions(
            self,
            preds_df: pl.DataFrame,
            iteration: int,
            clf_name: str,
            threshold: float,
        ) -> None:
            """
            Save predictions above threshold to a parquet file.
            Columns:
            filename, prob_real, pred_class, classifier, iteration, threshold
            """
            if preds_df is None or "prob_real" not in preds_df.columns:
                return

            out_dir = self.out_path / f"run_{iteration}" / "selected_preds"
            out_dir.mkdir(exist_ok=True)

            flagged = preds_df.filter(preds_df["prob_real"] >= threshold)
            if flagged.is_empty():
                return

            flagged = flagged.with_columns([
                pl.lit(clf_name).alias("classifier"),
                pl.lit(iteration).alias("iteration"),
                pl.lit(float(threshold)).alias("threshold"),
            ])

            out_file = out_dir / f"{clf_name}_run{iteration}.parquet"
            flagged.write_parquet(out_file)

            # Log
            self.logger.info(
                f"[RUN {iteration}] Saved {len(flagged)} flagged sims for {clf_name} → {out_file}"
            )

    # =================================================================
    #                       ROC DATA EXPORT
    # =================================================================
    def _calculate_roc_data(self, preds_df: pl.DataFrame, true_labels: dict[str, int]) -> pl.DataFrame:
        """
        Calculate ROC curve data (FPR, TPR, thresholds) from predictions.
        
        Parameters
        ----------
        preds_df : pl.DataFrame
            DataFrame with columns: filename, prob_real
        true_labels : dict[str, int]
            Dictionary mapping filename to true label (0=real, 1=simulated)
        
        Returns
        -------
        pl.DataFrame
            DataFrame with columns: fpr, tpr, threshold
        """
        if preds_df.is_empty() or "prob_real" not in preds_df.columns:
            return pl.DataFrame({"fpr": [], "tpr": [], "threshold": []})
        
        # Get true labels for each prediction
        y_true = []
        y_score = []
        
        for row in preds_df.iter_rows(named=True):
            filename = row["filename"]
            if filename in true_labels:
                y_true.append(true_labels[filename])
                y_score.append(row["prob_real"])
        
        if len(y_true) == 0:
            return pl.DataFrame({"fpr": [], "tpr": [], "threshold": []})
        
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # Check if we have both classes
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            self.logger.warning(
                f"Cannot calculate ROC curve: only {len(unique_labels)} class(es) found. "
                "ROC requires both positive and negative classes."
            )
            return pl.DataFrame({"fpr": [], "tpr": [], "threshold": []})
        
        # Calculate ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=LABEL_SIMULATED)
            # Add point at (0,0) for completeness
            fpr = np.concatenate([[0.0], fpr])
            tpr = np.concatenate([[0.0], tpr])
            thresholds = np.concatenate([[1.0], thresholds])
            
            roc_df = pl.DataFrame({
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "threshold": thresholds.tolist(),
            })
            
            return roc_df
        except Exception as e:
            self.logger.warning(f"Failed to calculate ROC curve: {e}")
            return pl.DataFrame({"fpr": [], "tpr": [], "threshold": []})

    def _find_optimal_threshold_roc(self, preds_df: pl.DataFrame, true_labels: dict[str, int]) -> float:
        """
        Find optimal threshold using Youden's J statistic (maximizes TPR - FPR).
        
        Parameters
        ----------
        preds_df : pl.DataFrame
            DataFrame with columns: filename, prob_real
        true_labels : dict[str, int]
            Dictionary mapping filename to true label
        
        Returns
        -------
        float
            Optimal threshold value
        """
        roc_df = self._calculate_roc_data(preds_df, true_labels)
        
        if roc_df.is_empty():
            return 0.5  # Default threshold
        
        # Calculate Youden's J statistic (TPR - FPR) for each threshold
        roc_df = roc_df.with_columns([
            (pl.col("tpr") - pl.col("fpr")).alias("youden_j")
        ])
        
        # Find threshold with maximum Youden's J
        max_j_row = roc_df.filter(pl.col("youden_j") == roc_df["youden_j"].max())
        
        if max_j_row.height > 0:
            optimal_threshold = max_j_row["threshold"][0]
            return float(optimal_threshold)
        
        return 0.5  # Default threshold

    def _export_roc_data(self, preds_df: pl.DataFrame, clf_name: str, iteration: int, true_labels: dict[str, int]) -> None:
        """
        Export ROC curve data to CSV file.
        
        Parameters
        ----------
        preds_df : pl.DataFrame
            DataFrame with predictions
        clf_name : str
            Classifier name
        iteration : int
            Iteration number (1 or 2)
        true_labels : dict[str, int]
            Dictionary mapping filename to true label
        """
        roc_df = self._calculate_roc_data(preds_df, true_labels)
        
        if roc_df.is_empty():
            self.logger.warning(f"[RUN {iteration}] No ROC data to export for {clf_name}")
            return
        
        # Add classifier and iteration columns
        roc_df = roc_df.with_columns([
            pl.lit(clf_name).alias("classifier"),
            pl.lit(iteration).alias("iteration"),
        ])
        
        # Export to CSV
        out_dir = self.out_path / f"run_{iteration}" / "roc_data"
        out_dir.mkdir(exist_ok=True)
        
        csv_file = out_dir / f"{clf_name}_roc.csv"
        roc_df.write_csv(csv_file)
        
        self.logger.info(f"[RUN {iteration}] ROC data exported for {clf_name} → {csv_file}")

    # =================================================================
    #                    BEST MODEL SELECTION
    # =================================================================
    def _select_best_model(self, clf_name: str, run_dir: Path) -> Path | None:
        """
        Select the best model based on minimum validation loss from train_history.
        
        Parameters
        ----------
        clf_name : str
            Classifier name
        run_dir : Path
            Directory containing the classifier's output
        
        Returns
        -------
        Path | None
            Path to the best model checkpoint, or None if not found
        """
        train_history_path = run_dir / "train_history.parquet"
        
        if not train_history_path.exists():
            self.logger.warning(f"No train_history.parquet found for {clf_name} in {run_dir}")
            return None
        
        try:
            history_df = pl.read_parquet(train_history_path)
            
            if history_df.is_empty() or "val_loss" not in history_df.columns:
                self.logger.warning(f"Invalid train_history for {clf_name}")
                return None
            
            # Find epoch with minimum validation loss
            min_val_loss_row = history_df.filter(
                pl.col("val_loss") == history_df["val_loss"].min()
            )
            
            if min_val_loss_row.height == 0:
                self.logger.warning(f"Could not find minimum val_loss for {clf_name}")
                return None
            
            best_epoch = min_val_loss_row["epoch"][0]
            
            # Check if best_model.pt exists (saved during training)
            best_model_path = run_dir / "best_model.pt"
            if best_model_path.exists():
                self.logger.info(f"[BEST MODEL] {clf_name}: using best_model.pt (epoch {best_epoch})")
                return best_model_path
            
            # Check checkpoint directory
            checkpoint_dir = run_dir / "checkpoint"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("best_model_*.pt"))
                if checkpoint_files:
                    best_checkpoint = sorted(checkpoint_files)[-1]  # Get most recent
                    self.logger.info(f"[BEST MODEL] {clf_name}: using {best_checkpoint.name} (epoch {best_epoch})")
                    return best_checkpoint
            
            self.logger.warning(f"No model checkpoint found for {clf_name} at epoch {best_epoch}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error selecting best model for {clf_name}: {e}")
            return None

    # =================================================================
    #       SINGLE RUN = TRAIN (all) → PREDICT (compatibles) → FILTER
    # =================================================================
    def run_single_iteration(self, iteration: int, threshold: float):
        """
        Run one iteration:
            - train all classifiers
            - predict on simulated alignments for supported ones
            - select simulated alignments with prob_real ≥ threshold
            - return both selected filenames AND full prediction dataframe
        """
        iter_dir = self.out_path / f"run_{iteration}"
        iter_dir.mkdir(exist_ok=True)

        selected: set[str] = set()
        preds_all = []   # <--- PATCH

        # ---------------- RUN2 SAFETY ----------------
        one_class_only = (iteration == 2 and not self._has_two_classes())
        if one_class_only:
            self.logger.warning(
                f"[RUN {iteration}] Only one class detected — deep models will be skipped."
            )

        # -------- TRAIN ----------
        self.logger.info(
            f"────────────────────────────────────────────\n"
            f" RUN {iteration} — TRAINING ({len(self.classifiers)} classifiers)\n"
            f"────────────────────────────────────────────"
        )

        for clf in tqdm(self.classifiers, desc=f"[RUN {iteration}] Training", unit="clf"):
            clf_name = clf["classifier"]
            out_dir = iter_dir / clf["out_dir"]
            out_dir.mkdir(parents=True, exist_ok=True)

            # Skip deep models in RUN2 if only 1 class
            if one_class_only and clf["is_deep"] and clf_name != "AACnnClassifier":
                self.logger.info(
                    f"[RUN {iteration}] Skipping {clf_name} (deep model — single class)"
                )
                continue

            # Skip Logistic Regression if one class
            if clf_name == "LogisticRegressionClassifier" and one_class_only:
                self.logger.info(
                    f"[RUN {iteration}] Skipping LogisticRegressionClassifier (single class)"
                )
                continue

            self.train_classifier(clf, out_dir)

        # -------- PREDICT ----------
        self.logger.info(
            f"────────────────────────────────────────────\n"
            f" RUN {iteration} — PREDICTIONS\n"
            f"────────────────────────────────────────────"
        )

        for clf in tqdm(self.classifiers, desc=f"[RUN {iteration}] Predictions", unit="clf"):
            clf_name = clf["classifier"]
            clf_dir = iter_dir / clf["out_dir"]

            if one_class_only and clf["is_deep"] and clf_name != "AACnnClassifier":
                continue

            if clf_name == "LogisticRegressionClassifier" and one_class_only:
                continue

            preds = self.predict_on_sim(clf, clf_dir)

            if preds is None:
                continue

            # -------- Export ROC data --------
            if "prob_real" in preds.columns and hasattr(self.base_data, 'labels'):
                self._export_roc_data(preds, clf_name, iteration, self.base_data.labels)

            # -------- PATCH: concat predictions --------
            preds_all.append(preds)   # <---

            # -------- FILTERING with optimal threshold --------
            if "prob_real" in preds.columns:
                # Use optimal threshold from ROC instead of fixed threshold
                optimal_threshold = self._find_optimal_threshold_roc(preds, self.base_data.labels)
                self.logger.info(
                    f"[RUN {iteration}] {clf_name}: optimal threshold = {optimal_threshold:.4f} (from ROC)"
                )
                # Filter: only keep simulations (LABEL_SIMULATED = 1) that are flagged as real
                # First, add true labels to predictions for filtering
                if hasattr(self.base_data, 'labels'):
                    # Create a DataFrame with labels for joining
                    labels_df = pl.DataFrame({
                        "filename": list(self.base_data.labels.keys()),
                        "true_label": list(self.base_data.labels.values())
                    })
                    # Join to add labels to predictions
                    preds_with_labels = preds.join(labels_df, on="filename", how="left")
                    # Keep only simulations (true_label == LABEL_SIMULATED) with prob_real >= threshold
                    flagged = preds_with_labels.filter(
                        (pl.col("true_label") == LABEL_SIMULATED) & 
                        (pl.col("prob_real") >= optimal_threshold)
                    )
                else:
                    # Fallback: if no labels, just filter by threshold (old behavior)
                    flagged = preds.filter(preds["prob_real"] >= optimal_threshold)
                for fname in flagged["filename"]:
                    selected.add(fname)

        # Merge all predictions into ONE DataFrame
        if preds_all:
            preds_all = pl.concat(preds_all)
        else:
            preds_all = pl.DataFrame({"filename": [], "prob_real": [], "pred_class": []})

        total = len(self.base_data.source_simulated.files)
        frac = len(selected) / total if total > 0 else 0.0

        self.logger.info(f"[RUN {iteration}] Selected {len(selected)} sims ({frac:.2%})")

        return list(selected), preds_all   # <--- PATCH



    # =================================================================
    #                BUILD DATASET FOR RUN2
    # =================================================================
    def build_run2_dataset(self, selected: list[str]) -> None:
        """
        Build the dataset for Run2:
            - copy all real alignments
            - copy only selected simulated alignments
        """
        r2_real = self.out_path / "run_2_real"
        r2_sim = self.out_path / "run_2_sim"

        r2_real.mkdir(exist_ok=True)
        r2_sim.mkdir(exist_ok=True)

        # Copy real
        for f in tqdm(self.base_data.source_real.files, desc="[RUN 2] Copy REAL", unit="file"):
            shutil.copy(f, r2_real / f.name)

        # Copy selected sim
        for fname in tqdm(selected, desc="[RUN 2] Copy SIM", unit="file"):
            src = self.base_data.source_simulated.root / fname
            if src.exists():
                shutil.copy(src, r2_sim / fname)

    # =================================================================
    #                  RUN 2 ONLY WITH BEST MODELS
    # =================================================================
    def run2_only_with_best_models(self) -> None:
        """
        Run only Run 2, selecting the best model from each classifier based on 
        minimum validation loss from Run 1.
        
        Assumes Run 1 has already been executed and models are trained.
        """
        self.logger.info("=== RUN 2 ONLY (WITH BEST MODELS) ===")
        
        run1_dir = self.out_path / "run_1"
        run2_dir = self.out_path / "run_2"
        run2_dir.mkdir(exist_ok=True)
        
        # Select best models from Run 1
        best_models = {}
        for clf in self.classifiers:
            clf_name = clf["classifier"]
            clf_run1_dir = run1_dir / clf["out_dir"]
            
            if not clf_run1_dir.exists():
                self.logger.warning(f"[BEST MODEL] Run 1 directory not found for {clf_name}, skipping")
                continue
            
            best_model_path = self._select_best_model(clf_name, clf_run1_dir)
            if best_model_path:
                best_models[clf_name] = {
                    "model_path": best_model_path,
                    "clf": clf,
                    "run1_dir": clf_run1_dir,
                }
        
        if not best_models:
            self.logger.error("No best models found from Run 1. Cannot proceed with Run 2.")
            return
        
        self.logger.info(f"[BEST MODEL] Selected {len(best_models)} best models for Run 2")
        
        # Copy best models to Run 2 directories
        for clf_name, model_info in best_models.items():
            clf = model_info["clf"]
            run2_clf_dir = run2_dir / clf["out_dir"]
            run2_clf_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy best model
            best_model_dest = run2_clf_dir / "best_model.pt"
            shutil.copy(model_info["model_path"], best_model_dest)
            self.logger.info(f"[BEST MODEL] Copied best model for {clf_name} to Run 2")
            
            # Copy train_history if it exists
            train_history_src = model_info["run1_dir"] / "train_history.parquet"
            if train_history_src.exists():
                train_history_dest = run2_clf_dir / "train_history.parquet"
                shutil.copy(train_history_src, train_history_dest)
        
        # Now run predictions on Run 2 dataset using best models
        self._current_iteration = 2
        
        selected: set[str] = set()
        preds_all = []
        
        # Check if Run 2 dataset exists
        if not (self.out_path / "run_2_real").exists() or not (self.out_path / "run_2_sim").exists():
            self.logger.warning("Run 2 dataset not found. Using current base_data.")
        else:
            # Reload dataset for Run 2
            self.base_data = Data(
                source_real=FastaSource(self.out_path / "run_2_real"),
                source_simulated=FastaSource(self.out_path / "run_2_sim"),
                tokenizer=self.tokenizer,
            )
        
        # PREDICT using best models
        self.logger.info(
            f"────────────────────────────────────────────\n"
            f" RUN 2 — PREDICTIONS WITH BEST MODELS\n"
            f"────────────────────────────────────────────"
        )
        
        for clf_name, model_info in best_models.items():
            clf = model_info["clf"]
            clf_dir = run2_dir / clf["out_dir"]
            
            # Skip if single class and incompatible
            one_class_only = not self._has_two_classes()
            if one_class_only and clf["is_deep"] and clf_name != "AACnnClassifier":
                self.logger.info(f"[RUN 2] Skipping {clf_name} (deep model — single class)")
                continue
            if clf_name == "LogisticRegressionClassifier" and one_class_only:
                self.logger.info(f"[RUN 2] Skipping {clf_name} (single class)")
                continue
            
            preds = self.predict_on_sim(clf, clf_dir)
            
            if preds is None:
                continue
            
            # Export ROC data
            if "prob_real" in preds.columns and hasattr(self.base_data, 'labels'):
                self._export_roc_data(preds, clf_name, 2, self.base_data.labels)
            
            preds_all.append(preds)
            
            # Filtering with optimal threshold
            if "prob_real" in preds.columns:
                optimal_threshold = self._find_optimal_threshold_roc(preds, self.base_data.labels)
                self.logger.info(
                    f"[RUN 2] {clf_name}: optimal threshold = {optimal_threshold:.4f} (from ROC)"
                )
                # Filter: only keep simulations (LABEL_SIMULATED = 1) that are flagged as real
                # First, add true labels to predictions for filtering
                if hasattr(self.base_data, 'labels'):
                    # Create a DataFrame with labels for joining
                    labels_df = pl.DataFrame({
                        "filename": list(self.base_data.labels.keys()),
                        "true_label": list(self.base_data.labels.values())
                    })
                    # Join to add labels to predictions
                    preds_with_labels = preds.join(labels_df, on="filename", how="left")
                    # Keep only simulations (true_label == LABEL_SIMULATED) with prob_real >= threshold
                    flagged = preds_with_labels.filter(
                        (pl.col("true_label") == LABEL_SIMULATED) & 
                        (pl.col("prob_real") >= optimal_threshold)
                    )
                else:
                    # Fallback: if no labels, just filter by threshold (old behavior)
                    flagged = preds.filter(preds["prob_real"] >= optimal_threshold)
                for fname in flagged["filename"]:
                    selected.add(fname)
        
        # Merge all predictions
        if preds_all:
            preds_all = pl.concat(preds_all)
        else:
            preds_all = pl.DataFrame({"filename": [], "prob_real": [], "pred_class": []})
        
        # Save predictions
        preds_all.write_parquet(self.out_path / "run_2/preds_run2.parquet")
        
        self.logger.info(f"[RUN 2] Selected {len(selected)} sims flagged REAL")
        self.logger.info("=== RUN 2 COMPLETE ===")

    # =================================================================
    #                     TWO ITERATIONS (MAIN MODE)
    # =================================================================
    def run_two_iterations(self, threshold: float) -> None:
        """
        Full two-iteration pipeline:
            - Run1 on full dataset (real + all simulated)
            - Filter simulated alignments
            - Build Run2 dataset
            - Run2 on refined dataset
            - Save full prediction tables for both runs
        """

        # ---------------- RUN 1 ----------------
        self.logger.info("=== RUN 1 START ===")

        self._current_iteration = 1
        selected_1, preds_run1 = self.run_single_iteration(1, threshold)

        self.logger.info(f"[RUN 1] {len(selected_1)} sims flagged REAL")

        # Save full prediction DF
        preds_run1.write_parquet(self.out_path / "run_1/preds_run1.parquet")

        # ---------------- Build RUN2 dataset ----------------
        self.logger.info("=== BUILDING RUN 2 DATASET ===")
        self.build_run2_dataset(selected_1)

        # ---------------- Reload dataset ----------------
        self.base_data = Data(
            source_real=FastaSource(self.out_path / "run_2_real"),
            source_simulated=FastaSource(self.out_path / "run_2_sim"),
            tokenizer=self.tokenizer,
        )

        # ---------------- RUN 2 with best models ----------------
        self.logger.info("=== RUN 2 START (WITH BEST MODELS) ===")
        self.run2_only_with_best_models()

        self.logger.info("=== PIPELINE COMPLETE ===")



    def run(self) -> None:
        """Single-run mode (no filtering)."""
        self.logger.info("=== SINGLE RUN START ===")

        self._current_iteration = 1   # <-- par sécurité
        self.run_single_iteration(1, self.threshold)


# =====================================================================
#                               MAIN
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pipeline runner")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--two-iterations", action="store_true", help="Run both Run1 and Run2")
    parser.add_argument("--run2-only", action="store_true", help="Run only Run2 with best models from Run1")
    args = parser.parse_args()

    pipeline = Pipeline.from_config(
        args.config,
        progress_bar=not args.no_progress,
        disable_compile=args.no_compile,
        threshold=args.threshold,
    )

    if args.run2_only:
        pipeline.run2_only_with_best_models()
    elif args.two_iterations:
        pipeline.run_two_iterations(args.threshold)
    else:
        pipeline.run()
