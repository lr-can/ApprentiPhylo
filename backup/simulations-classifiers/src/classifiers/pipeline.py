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
import gc

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
        
        # Configuration for generating new simulations between run1 and run2
        self.sim_config_2: dict | None = None

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
        
        # Load sim_config_2 if provided (for generating new simulations between run1 and run2)
        if "sim_config_2" in cfg:
            pipeline.sim_config_2 = cfg["sim_config_2"]
            pipeline.logger.info("Loaded sim_config_2 for generating new simulations between run1 and run2")

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
                from torch.utils.data import DataLoader, random_split
                from torch import Generator
                from classifiers.utils import RANDOM_SEED
                
                # CRITICAL FIX: Apply train/validation split to prevent data leakage
                if model.dataset is None:
                    msg = "Dataset has not been defined"
                    raise ValueError(msg)
                
                model.logger.log("--- Creating loaders (with train/val split) ---")
                generator = Generator().manual_seed(RANDOM_SEED)
                train_dataset, valid_dataset = random_split(
                    model.dataset, 
                    [model.split_proportion, 1 - model.split_proportion], 
                    generator=generator
                )
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=model.batch_size,
                    shuffle=True,
                    collate_fn=collate_x_y,
                )
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=model.batch_size,
                    shuffle=False,
                    collate_fn=collate_x_y,
                )
                return train_loader, valid_loader

            model.get_loaders = patched_get_loaders
            self.logger.info(f"[RUN] Applied collate_x_y patch to {clf_name} (with train/val split)")

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
        # pos_label=LABEL_REAL because prob_real represents P(REAL)
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=LABEL_REAL)
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
        Find optimal threshold using ROC curve analysis.
        Uses Youden's J statistic (maximizes TPR - FPR).
        
        Parameters
        ----------
        preds_df : pl.DataFrame
            DataFrame with columns: filename, prob_real
        true_labels : dict[str, int]
            Dictionary mapping filename to true label
        
        Returns
        -------
        float
            Optimal threshold from ROC curve
        """
        # Build arrays for ROC calculation
        y_true = []
        y_score = []
        
        for row in preds_df.iter_rows(named=True):
            filename = row["filename"]
            if filename in true_labels:
                y_true.append(true_labels[filename])
                y_score.append(row["prob_real"])
        
        if len(y_true) == 0:
            return 0.5  # Default threshold
        
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # Check if we have both classes
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            self.logger.warning(
                f"Cannot calculate optimal threshold: only {len(unique_labels)} class(es) found. "
                "ROC requires both positive and negative classes."
            )
            return 0.5  # Default threshold
        
        # Calculate ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            
            # Check if AUC < 0.5 (model predicts inverse)
            inverted = False
            if auc < 0.5:
                self.logger.warning(
                    f"⚠️  AUC = {auc:.4f} < 0.5: Model predictions are inverted!"
                )
                self.logger.warning(
                    f"    Inverting predictions for threshold calculation"
                )
                self.logger.warning(
                    f"    This suggests labels may be swapped in training/prediction"
                )
                # Invert scores and recalculate ROC
                y_score_inverted = 1 - y_score
                fpr, tpr, thresholds = roc_curve(y_true, y_score_inverted)
                auc_inverted = roc_auc_score(y_true, y_score_inverted)
                self.logger.info(f"    After inversion: AUC = {auc_inverted:.4f}")
                inverted = True
            
            # Calculate Youden's J statistic (TPR - FPR)
            j_scores = tpr - fpr
            
            # Find index of maximum J
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = float(thresholds[optimal_idx])
            
            # If we inverted the scores, we need to invert the threshold back
            # for use with original scores
            if inverted:
                optimal_threshold = 1.0 - optimal_threshold
                self.logger.info(f"    Threshold for original scores: {optimal_threshold:.4f}")
            
            # Handle inf or invalid threshold
            if np.isinf(optimal_threshold) or np.isnan(optimal_threshold) or optimal_threshold < 0 or optimal_threshold > 1.0:
                self.logger.warning(f"Optimal threshold is invalid ({optimal_threshold}), using 0.5 instead")
                optimal_threshold = 0.5
            
            final_auc = auc_inverted if inverted else auc
            self.logger.info(f"ROC AUC (corrected): {final_auc:.4f}")
            self.logger.info(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")
            self.logger.info(f"  At threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}, J={j_scores[optimal_idx]:.4f}")
            
            return optimal_threshold
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal threshold: {e}")
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

    def _select_best_overall_model(self, run1_dir: Path) -> dict | None:
        """
        Select the single best model across all classifiers from Run 1.
        Comparison is based on F1 score (primary) and validation loss (secondary).
        
        Parameters
        ----------
        run1_dir : Path
            Directory containing Run 1 results
        
        Returns
        -------
        dict | None
            Dictionary with keys: 'clf_name', 'clf', 'run1_dir', 'model_path', 'val_loss', 'f1_score'
            or None if no valid model found
        """
        best_candidates = []
        
        for clf in self.classifiers:
            clf_name = clf["classifier"]
            clf_run1_dir = run1_dir / clf["out_dir"]
            
            if not clf_run1_dir.exists():
                continue
            
            train_history_path = clf_run1_dir / "train_history.parquet"
            if not train_history_path.exists():
                # For LogisticRegressionClassifier, check summary.json instead
                if clf_name == "LogisticRegressionClassifier":
                    summary_path = clf_run1_dir / "summary.json"
                    if summary_path.exists():
                        try:
                            import json
                            with open(summary_path, "r") as f:
                                summary = json.load(f)
                            # Extract average F1 score from fold scores
                            f1_scores = [float(s) for s in summary.get("fold_f1_scores", [])]
                            if f1_scores:
                                avg_f1 = np.mean(f1_scores)
                                best_candidates.append({
                                    "clf_name": clf_name,
                                    "clf": clf,
                                    "run1_dir": clf_run1_dir,
                                    "model_path": None,  # LR doesn't save .pt files
                                    "val_loss": None,
                                    "f1_score": avg_f1,
                                    "val_acc": np.mean([float(s) for s in summary.get("fold_accuracies", [])]) if summary.get("fold_accuracies") else None,
                                })
                        except Exception as e:
                            self.logger.warning(f"Error reading summary for {clf_name}: {e}")
                continue
            
            try:
                history_df = pl.read_parquet(train_history_path)
                
                if history_df.is_empty():
                    continue
                
                # Get best metrics (minimum val_loss)
                if "val_loss" in history_df.columns:
                    min_val_loss_row = history_df.filter(
                        pl.col("val_loss") == history_df["val_loss"].min()
                    )
                    if min_val_loss_row.height > 0:
                        best_row = min_val_loss_row.row(0, named=True)
                        
                        model_path = self._select_best_model(clf_name, clf_run1_dir)
                        if model_path:
                            best_candidates.append({
                                "clf_name": clf_name,
                                "clf": clf,
                                "run1_dir": clf_run1_dir,
                                "model_path": model_path,
                                "val_loss": best_row.get("val_loss"),
                                "f1_score": best_row.get("f1_score", best_row.get("f1", None)),
                                "val_acc": best_row.get("val_acc", None),
                            })
            except Exception as e:
                self.logger.warning(f"Error processing {clf_name} for best model selection: {e}")
                continue
        
        if not best_candidates:
            self.logger.error("No valid models found for best model selection")
            return None
        
        # Select best: prioritize F1 score, then val_loss (lower is better)
        # Filter out None f1_scores
        valid_candidates = [c for c in best_candidates if c["f1_score"] is not None]
        
        if not valid_candidates:
            self.logger.error("No valid models with F1 scores found")
            return None
        
        # Sort by F1 score (descending), then by val_loss (ascending if available)
        def sort_key(c):
            f1 = c["f1_score"] if c["f1_score"] is not None else 0.0
            val_loss = c["val_loss"] if c["val_loss"] is not None else float('inf')
            return (-f1, val_loss)
        
        best_candidates.sort(key=sort_key)
        best = best_candidates[0]
        
        # Format val_loss for logging
        val_loss_str = f"{best['val_loss']:.4f}" if best['val_loss'] is not None else "N/A"
        
        self.logger.info(
            f"[BEST MODEL] Selected {best['clf_name']} as best overall model "
            f"(F1={best['f1_score']:.4f}, val_loss={val_loss_str})"
        )
        
        return best

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

            # -------- FILTERING with optimal threshold from ROC --------
            if "prob_real" in preds.columns:
                # Calculate optimal threshold using Youden's J statistic
                optimal_threshold = self._find_optimal_threshold_roc(preds, self.base_data.labels)
                self.logger.info(
                    f"[RUN {iteration}] {clf_name}: optimal threshold = {optimal_threshold:.4f} (from ROC - Youden's J)"
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
    #          GENERATE AND FILTER NEW SIMULATIONS FOR RUN2
    # =================================================================
    def generate_and_filter_new_simulations(
        self, 
        best_model_info: dict, 
        initial_sim_count: int, 
        selected_count: int,
        threshold: float
    ) -> tuple[list[str], pl.DataFrame]:
        """
        Generate new simulations to balance the dataset size for Run2, then filter them
        using the best model from Run1.
        
        Parameters
        ----------
        best_model_info : dict
            Dictionary containing best model information from _select_best_overall_model()
        initial_sim_count : int
            Initial number of simulated alignments
        selected_count : int
            Number of simulations selected from Run1
        threshold : float
            Threshold for filtering new simulations
        
        Returns
        -------
        tuple[list[str], pl.DataFrame]
            Selected filenames and full predictions DataFrame for new simulations
        """
        if self.sim_config_2 is None:
            self.logger.info("[NEW SIMS] No sim_config_2 provided, skipping generation of new simulations")
            return [], pl.DataFrame()
        
        # Calculate how many new simulations we need
        # Goal: have the same number of simulated data as real data
        num_real = len(self.base_data.source_real.files)
        num_needed = max(0, num_real - selected_count)
        
        if num_needed == 0:
            self.logger.info(
                f"[NEW SIMS] No new simulations needed. "
                f"Real: {num_real}, Selected from Run1: {selected_count} (already balanced)"
            )
            return [], pl.DataFrame()
        
        self.logger.info(
            f"[NEW SIMS] Generating {num_needed} new simulations to balance dataset "
            f"(Real: {num_real}, Selected from Run1: {selected_count}, Needed: {num_needed})"
        )
        
        # Setup directories
        new_sim_dir = self.out_path / "run_2" / "new_sim"
        new_sim_dir.mkdir(parents=True, exist_ok=True)
        
        # Import BppSimulator (need to handle import path)
        # The pipeline is in backup/simulations-classifiers/src/classifiers/
        # We need to go up to project root and then to scripts/
        try:
            import sys
            # Calculate path: from backup/simulations-classifiers/src/classifiers/ to project root
            # __file__ is at: .../backup/simulations-classifiers/src/classifiers/pipeline.py
            # We need: .../scripts/simulation.py
            pipeline_file = Path(__file__).resolve()
            # Go up: classifiers -> src -> simulations-classifiers -> backup -> project_root
            project_root = pipeline_file.parents[4]  # backup/simulations-classifiers/src/classifiers -> project_root
            scripts_path = project_root / "scripts"
            
            if not scripts_path.exists():
                # Alternative: try going up one more level if structure is different
                project_root = pipeline_file.parents[5]
                scripts_path = project_root / "scripts"
            
            if scripts_path.exists():
                if str(scripts_path) not in sys.path:
                    sys.path.insert(0, str(scripts_path))
                from simulation import BppSimulator
            else:
                raise ImportError(f"Scripts directory not found at {scripts_path}")
        except ImportError as e:
            self.logger.error(f"[NEW SIMS] Could not import BppSimulator: {e}")
            self.logger.error(f"[NEW SIMS] Tried path: {scripts_path}")
            return [], pl.DataFrame()
        
        # Extract simulation config
        sim_config = self.sim_config_2
        config_file = Path(sim_config.get("config"))
        tree_dir = Path(sim_config.get("tree"))
        alphabet = sim_config.get("alphabet", "aa")
        ext_rate = sim_config.get("ext_rate")
        
        # Determine source alignments directory (use run_2_real as source for new simulations)
        source_align_dir = self.out_path / "run_2_real"
        if not source_align_dir.exists():
            self.logger.warning(f"[NEW SIMS] Source alignment directory not found: {source_align_dir}")
            return [], pl.DataFrame()
        
        # Setup for generation and filtering
        clf_name = best_model_info["clf_name"]
        clf = best_model_info["clf"]
        run1_dir = self.out_path / "run_1"
        clf_run1_dir = run1_dir / clf["out_dir"]
        
        from classifiers.data.sources import FastaSource
        from classifiers.data.data import Data
        
        # Generate a large batch of simulations (we'll filter and take exactly what we need)
        # Generate 4-5x the needed amount to account for filtering
        num_to_generate = max(num_needed * 4, 500)  # At least 500 simulations
        
        self.logger.info(
            f"[NEW SIMS] Generating {num_to_generate} simulations (will filter to get exactly {num_needed})"
        )
        
        # Create BppSimulator instance
        bpp_sim = BppSimulator(
            align=str(source_align_dir),
            tree=str(tree_dir),
            config=str(config_file),
            output=str(new_sim_dir),
            ext_rate=ext_rate
        )
        
        # Generate simulations
        try:
            bpp_sim.simulate()
        except Exception as e:
            self.logger.error(f"[NEW SIMS] Error during simulation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [], pl.DataFrame()
        
        # Count generated files
        generated_files = list(new_sim_dir.glob("*.fasta"))
        num_generated = len(generated_files)
        self.logger.info(f"[NEW SIMS] Generated {num_generated} simulation files")
        
        if num_generated == 0:
            self.logger.warning("[NEW SIMS] No simulations were generated")
            return [], pl.DataFrame()
        
        # Now filter using the best model from Run1
        self.logger.info("[NEW SIMS] Filtering new simulations using best model from Run1...")
        
        # Create a temporary Data object with new simulations
        temp_real_dir = self.out_path / "run_2_real"
        temp_sim_dir = new_sim_dir
        
        temp_data = Data(
            source_real=FastaSource(temp_real_dir),
            source_simulated=FastaSource(temp_sim_dir),
            tokenizer=self.tokenizer,
        )
        
        # Save original base_data and temporarily replace it
        original_base_data = self.base_data
        self.base_data = temp_data
        
        # Make predictions using the best model from Run1
        original_iteration = getattr(self, "_current_iteration", None)
        self._current_iteration = 1
        preds = self.predict_on_sim(clf, clf_run1_dir)
        if original_iteration is not None:
            self._current_iteration = original_iteration
        
        # Restore original base_data
        self.base_data = original_base_data
        
        if preds is None or preds.is_empty():
            self.logger.warning("[NEW SIMS] No predictions generated for new simulations")
            return [], pl.DataFrame()
        
        if "prob_real" not in preds.columns:
            self.logger.warning("[NEW SIMS] prob_real column not found in predictions")
            return [], pl.DataFrame()
        
        # Calculate optimal threshold using Youden's J statistic (like in Run1)
        # This is the same method used for filtering in Run1
        if hasattr(temp_data, 'labels') and temp_data.labels:
            optimal_threshold = self._find_optimal_threshold_roc(preds, temp_data.labels)
            self.logger.info(
                f"[NEW SIMS] Optimal threshold (Youden's J) = {optimal_threshold:.4f} "
                f"(same method as Run1 filtering)"
            )
        else:
            # Fallback to provided threshold if no labels available
            optimal_threshold = threshold
            self.logger.info(
                f"[NEW SIMS] Using provided threshold = {optimal_threshold:.4f} "
                f"(labels not available for Youden calculation)"
            )
        
        # Filter predictions using optimal threshold (same as Run1)
        filtered = preds.filter(pl.col("prob_real") >= optimal_threshold)
        selected_filenames = filtered["filename"].to_list()
        
        self.logger.info(
            f"[NEW SIMS] Filtered {len(selected_filenames)}/{num_generated} new simulations "
            f"(threshold={optimal_threshold:.4f})"
        )
        
        # If we don't have enough, we need to generate more
        if len(selected_filenames) < num_needed:
            self.logger.warning(
                f"[NEW SIMS] Only {len(selected_filenames)}/{num_needed} simulations passed filtering. "
                f"Generating additional simulations..."
            )
            
            # Generate additional simulations in batches until we have enough
            batch_num = 0
            max_additional_batches = 5
            
            while len(selected_filenames) < num_needed and batch_num < max_additional_batches:
                batch_num += 1
                # Create a temporary batch directory
                batch_dir = new_sim_dir / f"batch_{batch_num}"
                batch_dir.mkdir(exist_ok=True)
                
                self.logger.info(f"[NEW SIMS] Additional batch {batch_num}: Generating more simulations...")
                
                batch_sim = BppSimulator(
                    align=str(source_align_dir),
                    tree=str(tree_dir),
                    config=str(config_file),
                    output=str(batch_dir),
                    ext_rate=ext_rate
                )
                
                try:
                    batch_sim.simulate()
                except Exception as e:
                    self.logger.error(f"[NEW SIMS] Error in additional batch {batch_num}: {e}")
                    break
                
                # Move batch files to main directory with unique names
                batch_files = list(batch_dir.glob("*.fasta"))
                for bf in batch_files:
                    # Add batch number to filename to avoid conflicts
                    new_name = f"batch{batch_num}_{bf.name}"
                    bf.rename(new_sim_dir / new_name)
                
                # Re-predict on all files including new batch
                temp_data_batch = Data(
                    source_real=FastaSource(temp_real_dir),
                    source_simulated=FastaSource(new_sim_dir),
                    tokenizer=self.tokenizer,
                )
                
                # Use original_base_data (saved at the beginning) for restoration
                self.base_data = temp_data_batch
                
                original_iteration = getattr(self, "_current_iteration", None)
                self._current_iteration = 1
                batch_preds = self.predict_on_sim(clf, clf_run1_dir)
                if original_iteration is not None:
                    self._current_iteration = original_iteration
                
                # Restore to original_base_data (not temp_data)
                self.base_data = original_base_data
                
                if batch_preds is not None and not batch_preds.is_empty() and "prob_real" in batch_preds.columns:
                    # Recalculate threshold with all data
                    if hasattr(temp_data_batch, 'labels') and temp_data_batch.labels:
                        optimal_threshold = self._find_optimal_threshold_roc(batch_preds, temp_data_batch.labels)
                    
                    filtered_batch = batch_preds.filter(pl.col("prob_real") >= optimal_threshold)
                    batch_selected = [f for f in filtered_batch["filename"].to_list() if f not in selected_filenames]
                    selected_filenames.extend(batch_selected)
                    
                    self.logger.info(
                        f"[NEW SIMS] Batch {batch_num}: Added {len(batch_selected)} more filtered simulations. "
                        f"Total: {len(selected_filenames)}/{num_needed}"
                    )
                    
                    preds = batch_preds  # Update predictions
                
                # Clean up batch directory
                import shutil
                if batch_dir.exists():
                    shutil.rmtree(batch_dir)
        
        # Take exactly the number needed
        if len(selected_filenames) > num_needed:
            import random
            selected_filenames = random.sample(selected_filenames, num_needed)
            self.logger.info(
                f"[NEW SIMS] Randomly sampled {num_needed} from {len(selected_filenames)} filtered simulations"
            )
        elif len(selected_filenames) < num_needed:
            self.logger.warning(
                f"[NEW SIMS] Only {len(selected_filenames)}/{num_needed} simulations passed filtering. "
                f"Dataset will be unbalanced."
            )
        
        # Save predictions for dashboard
        preds.write_parquet(self.out_path / "run_2" / "new_sim_predictions.parquet")
        
        return selected_filenames, preds

    # =================================================================
    #                BUILD DATASET FOR RUN2
    # =================================================================
    def build_run2_dataset(self, selected: list[str], new_sim_selected: list[str] = None) -> None:
        """
        Build the dataset for Run2:
            - copy all real alignments
            - copy only selected simulated alignments from Run1
            - add newly generated and filtered simulations (if any)
        
        Parameters
        ----------
        selected : list[str]
            Filenames of simulations selected from Run1
        new_sim_selected : list[str], optional
            Filenames of newly generated simulations that passed filtering
        """
        r2_real = self.out_path / "run_2_real"
        r2_sim = self.out_path / "run_2_sim"
        new_sim_dir = self.out_path / "run_2" / "new_sim"

        r2_real.mkdir(exist_ok=True)
        r2_sim.mkdir(exist_ok=True)

        # Copy real
        for f in tqdm(self.base_data.source_real.files, desc="[RUN 2] Copy REAL", unit="file"):
            shutil.copy(f, r2_real / f.name)

        # Copy selected sim from Run1
        for fname in tqdm(selected, desc="[RUN 2] Copy SIM (from Run1)", unit="file"):
            # Remove "_sim" suffix if present (added during preprocessing for duplicate keys)
            # to get the original filename in the source directory
            # The suffix "_sim" is added to the stem (before extension), not the full filename
            if fname.endswith(".fasta"):
                stem = fname[:-6]  # Remove .fasta
                ext = ".fasta"
            elif fname.endswith(".fa"):
                stem = fname[:-3]  # Remove .fa
                ext = ".fa"
            else:
                stem = fname
                ext = ""
            
            # Try original filename (with _sim removed from stem if present)
            if stem.endswith("_sim"):
                original_fname = stem[:-4] + ext
            else:
                original_fname = fname
            
            # Try to find the file with original name first
            src = self.base_data.source_simulated.root / original_fname
            if not src.exists():
                # Fallback: try with the fname as-is (in case it wasn't renamed)
                src = self.base_data.source_simulated.root / fname
            
            if src.exists():
                # Copy with original filename (without _sim suffix) to avoid issues in RUN 2
                shutil.copy(src, r2_sim / original_fname)
            else:
                self.logger.warning(f"[RUN 2] File not found: {fname} (tried {original_fname} and {fname})")
        
        # Copy newly generated and filtered simulations
        # Track files already copied to avoid duplicates
        copied_files = set()
        for fname in selected:
            # Get the actual filename that was copied (without _sim suffix)
            if fname.endswith(".fasta"):
                stem = fname[:-6]
                ext = ".fasta"
            elif fname.endswith(".fa"):
                stem = fname[:-3]
                ext = ".fa"
            else:
                stem = fname
                ext = ""
            
            if stem.endswith("_sim"):
                copied_files.add(stem[:-4] + ext)
            else:
                copied_files.add(fname)
        
        if new_sim_selected:
            self.logger.info(f"[RUN 2] Adding {len(new_sim_selected)} newly generated simulations")
            copied_count = 0
            skipped_duplicates = 0
            
            for fname in tqdm(new_sim_selected, desc="[RUN 2] Copy NEW SIM", unit="file"):
                # Remove "_sim" suffix if present (added during preprocessing for duplicate keys)
                if fname.endswith(".fasta"):
                    stem = fname[:-6]  # Remove .fasta
                    ext = ".fasta"
                elif fname.endswith(".fa"):
                    stem = fname[:-3]  # Remove .fa
                    ext = ".fa"
                else:
                    stem = fname
                    ext = ""
                
                # Remove _sim suffix from stem if present
                if stem.endswith("_sim"):
                    original_fname = stem[:-4] + ext
                else:
                    original_fname = fname
                
                # Check if this file already exists (from Run1)
                if original_fname in copied_files:
                    # Add a unique prefix to avoid overwriting
                    dest_name = f"new_{original_fname}"
                    skipped_duplicates += 1
                else:
                    dest_name = original_fname
                
                # Try to find the file with original name (without _sim)
                src = new_sim_dir / original_fname
                if not src.exists():
                    # Fallback: try with the fname as-is (in case it wasn't renamed)
                    src = new_sim_dir / fname
                
                if src.exists():
                    # Copy with appropriate name (with prefix if duplicate)
                    shutil.copy(src, r2_sim / dest_name)
                    copied_files.add(dest_name)
                    copied_count += 1
                else:
                    self.logger.warning(f"[RUN 2] New sim file not found: {fname} (tried {original_fname} and {fname})")
            
            if skipped_duplicates > 0:
                self.logger.info(
                    f"[RUN 2] {skipped_duplicates} new simulations had duplicate names with Run1 files, "
                    f"prefixed with 'new_' to avoid overwriting"
                )
        
        # Log final counts (count actual files, not what we tried to copy)
        num_real_final = len(list(r2_real.glob("*.fasta"))) if r2_real.exists() else 0
        num_sim_final = len(list(r2_sim.glob("*.fasta"))) if r2_sim.exists() else 0
        
        # Count unique files actually copied
        actual_run1_count = len([f for f in r2_sim.glob("*.fasta") if not f.name.startswith("new_") and not f.name.startswith("batch")])
        actual_new_count = len([f for f in r2_sim.glob("*.fasta") if f.name.startswith("new_") or f.name.startswith("batch")])
        
        self.logger.info(
            f"[RUN 2] Dataset built: {num_real_final} real, {num_sim_final} simulated "
            f"({actual_run1_count} from Run1, {actual_new_count} newly generated)"
        )
        
        # Warn if counts don't match expected
        expected_sim = len(selected) + (len(new_sim_selected) if new_sim_selected else 0)
        if num_sim_final != expected_sim:
            self.logger.warning(
                f"[RUN 2] Mismatch: expected {expected_sim} simulations but found {num_sim_final} files. "
                f"Some files may have been overwritten or not copied."
            )
        
        # Warn if not balanced with real data
        if num_sim_final != num_real_final:
            self.logger.warning(
                f"[RUN 2] Dataset unbalanced: {num_real_final} real vs {num_sim_final} simulated "
                f"(difference: {num_real_final - num_sim_final})"
            )

    # =================================================================
    #                  RUN 2 RETRAIN BEST MODEL
    # =================================================================
    def run2_retrain_best_model(self, best_model_info: dict, threshold: float, initial_sim_count: int) -> tuple[set[str], pl.DataFrame]:
        """
        Retrain the best model from Run 1 with Run 2 dataset and make predictions.
        
        Parameters
        ----------
        best_model_info : dict
            Dictionary containing best model information from _select_best_overall_model()
        threshold : float
            Threshold for filtering predictions
        initial_sim_count : int
            Initial number of simulated alignments (for calculating retention proportion)
        
        Returns
        -------
        tuple[set[str], pl.DataFrame]
            Selected filenames and full predictions DataFrame
        """
        self.logger.info("=== RUN 2 START (RETRAINING BEST MODEL) ===")
        
        clf_name = best_model_info["clf_name"]
        clf = best_model_info["clf"]
        run2_dir = self.out_path / "run_2"
        run2_dir.mkdir(exist_ok=True)
        clf_run2_dir = run2_dir / clf["out_dir"]
        clf_run2_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"[RUN 2] Retraining {clf_name} with Run 2 dataset...")
        
        # Check if we need to skip due to single class
        one_class_only = not self._has_two_classes()
        if one_class_only:
            if clf["is_deep"] and clf_name != "AACnnClassifier":
                self.logger.warning(f"[RUN 2] Cannot retrain {clf_name} (deep model — single class)")
                return set(), pl.DataFrame()
            if clf_name == "LogisticRegressionClassifier":
                self.logger.warning(f"[RUN 2] Cannot retrain {clf_name} (single class)")
                return set(), pl.DataFrame()
        
        # ============================================================
        # CLEAN RESET: Remove all existing models and checkpoints
        # ============================================================
        # Remove existing best_model.pt
        existing_model = clf_run2_dir / "best_model.pt"
        if existing_model.exists():
            existing_model.unlink()
            self.logger.info(f"[RUN 2] Removed existing best_model.pt")
        
        # Remove checkpoint directory and all its contents
        checkpoint_dir = clf_run2_dir / "checkpoint"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            self.logger.info(f"[RUN 2] Removed checkpoint directory")
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info(f"[RUN 2] Cleared CUDA cache")
        
        # Force garbage collection
        gc.collect()
        
        self.logger.info(f"[RUN 2] Starting fresh training (new model instance, no weights from RUN 1)")
        
        # Train the model with Run 2 dataset (fresh initialization)
        # Note: train_classifier() creates a NEW model instance with random init
        # Seeds are controlled by RANDOM_SEED for reproducibility
        self._current_iteration = 2
        self.train_classifier(clf, clf_run2_dir)
        
        # Now predict with the retrained model
        self.logger.info(
            f"────────────────────────────────────────────\n"
            f" RUN 2 — PREDICTIONS (RETRAINED MODEL)\n"
            f"────────────────────────────────────────────"
        )
        
        preds = self.predict_on_sim(clf, clf_run2_dir)
        
        if preds is None:
            self.logger.warning(f"[RUN 2] No predictions generated for {clf_name}")
            return set(), pl.DataFrame()
        
        # Export ROC data
        if "prob_real" in preds.columns and hasattr(self.base_data, 'labels'):
            self._export_roc_data(preds, clf_name, 2, self.base_data.labels)
        
        selected: set[str] = set()
        
        # Filtering with optimal threshold
        if "prob_real" in preds.columns:
            optimal_threshold = self._find_optimal_threshold_roc(preds, self.base_data.labels)
            self.logger.info(
                f"[RUN 2] {clf_name}: optimal threshold = {optimal_threshold:.4f} (from ROC - Youden's J)"
            )
            # Filter: only keep simulations (LABEL_SIMULATED = 1) that are flagged as real
            if hasattr(self.base_data, 'labels'):
                labels_df = pl.DataFrame({
                    "filename": list(self.base_data.labels.keys()),
                    "true_label": list(self.base_data.labels.values())
                })
                preds_with_labels = preds.join(labels_df, on="filename", how="left")
                flagged = preds_with_labels.filter(
                    (pl.col("true_label") == LABEL_SIMULATED) & 
                    (pl.col("prob_real") >= optimal_threshold)
                )
            else:
                flagged = preds.filter(preds["prob_real"] >= optimal_threshold)
            for fname in flagged["filename"]:
                selected.add(fname)
        
        # Calculate retention proportion
        final_count = len(selected)
        retention_proportion = final_count / initial_sim_count if initial_sim_count > 0 else 0.0
        self.logger.info(
            f"[RUN 2] Selected {final_count} sims flagged REAL "
            f"({retention_proportion:.2%} of initial {initial_sim_count} simulated alignments)"
        )
        
        return selected, preds

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
                    f"[RUN 2] {clf_name}: optimal threshold = {optimal_threshold:.4f} (from ROC - Youden's J)"
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
            - Select best model from Run 1
            - Retrain best model with Run2 dataset
            - Make predictions and calculate retention proportion
            - Save full prediction tables for both runs
        """

        # ---------------- RUN 1 ----------------
        self.logger.info("=== RUN 1 START ===")

        self._current_iteration = 1
        selected_1, preds_run1 = self.run_single_iteration(1, threshold)

        self.logger.info(f"[RUN 1] {len(selected_1)} sims flagged REAL")

        # Save full prediction DF
        preds_run1.write_parquet(self.out_path / "run_1/preds_run1.parquet")

        # Store initial count of simulated alignments
        initial_sim_count = len(self.base_data.source_simulated.files)

        # ---------------- Select best model from Run 1 ----------------
        self.logger.info("=== SELECTING BEST MODEL FROM RUN 1 ===")
        run1_dir = self.out_path / "run_1"
        best_model_info = self._select_best_overall_model(run1_dir)
        
        if best_model_info is None:
            self.logger.error("Failed to select best model from Run 1. Cannot proceed with Run 2.")
            return

        # ---------------- Generate new simulations (if needed) ----------------
        new_sim_selected = []
        new_sim_preds = pl.DataFrame()
        
        if self.sim_config_2 is not None:
            self.logger.info("=== GENERATING NEW SIMULATIONS FOR RUN2 ===")
            new_sim_selected, new_sim_preds = self.generate_and_filter_new_simulations(
                best_model_info=best_model_info,
                initial_sim_count=initial_sim_count,
                selected_count=len(selected_1),
                threshold=threshold
            )
            if new_sim_selected:
                self.logger.info(f"[NEW SIMS] {len(new_sim_selected)} new simulations will be added to Run2 dataset")
        
        # ---------------- Build RUN2 dataset ----------------
        self.logger.info("=== BUILDING RUN 2 DATASET ===")
        self.build_run2_dataset(selected_1, new_sim_selected=new_sim_selected)

        # ---------------- Reload dataset ----------------
        self.base_data = Data(
            source_real=FastaSource(self.out_path / "run_2_real"),
            source_simulated=FastaSource(self.out_path / "run_2_sim"),
            tokenizer=self.tokenizer,
        )

        # ---------------- RUN 2: Retrain best model ----------------
        selected_2, preds_run2 = self.run2_retrain_best_model(
            best_model_info, 
            threshold,
            initial_sim_count
        )

        # Save Run 2 predictions
        if not preds_run2.is_empty():
            preds_run2.write_parquet(self.out_path / "run_2/preds_run2.parquet")

        # Final summary
        self.logger.info("=== PIPELINE COMPLETE ===")
        self.logger.info(
            f"Final retention: {len(selected_2)}/{initial_sim_count} simulated alignments "
            f"({len(selected_2)/initial_sim_count:.2%})"
        )



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
