#!/usr/bin/env python3
"""
Build a clean dashboard dataset based on (path, md5) identity.

Workflow:
1) Consider the Run2 dataset sims: results/classification/run_2_sim/*.fasta
2) Compute md5 + absolute path for those files
3) Run predictions on those sims using:
   - Run1 best model (e.g. results/classification/run_1/AACnnClassifier/best_model.pt)
   - Run2 best model (e.g. results/classification/run_2/AACnnClassifier/best_model.pt)
4) Reconstruct "passed run2" using the Run2 Youden threshold (from pipeline log),
   then intersect by MD5 (robust to renames).
5) Save parquet for dashboard2.py consumption.

Output:
results/classification/run_2/dashboard_dataset.parquet
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIF_DIR = PROJECT_ROOT / "results" / "classification"


def md5sum_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_latest_pipeline_log() -> Path | None:
    logs = sorted(CLASSIF_DIR.glob("pipeline_*.log"))
    return logs[-1] if logs else None


def read_youden_threshold(run_number: int, log_path: Path) -> float | None:
    """
    Extract optimal threshold from logs.
    RUN 1: matches "optimal threshold = X"
    RUN 2: matches "[RUN 2] ... optimal threshold = X"
    """
    content = log_path.read_text(errors="ignore")
    if run_number == 1:
        m = re.search(r"\[RUN 1\].*?optimal threshold = ([\d.]+)", content)
        if not m:
            # fallback older pattern
            m = re.search(r"optimal threshold = ([\d.]+)", content)
    else:
        m = re.search(r"\[RUN 2\].*?optimal threshold = ([\d.]+)", content)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def pick_best_classifier(run_dir: Path) -> str:
    """
    Pick best classifier from summary.json by max f1_score, tie-breaker by val_loss (min).
    """
    candidates: list[tuple[str, float, float]] = []
    for clf_dir in run_dir.iterdir():
        if not clf_dir.is_dir():
            continue
        summary = clf_dir / "summary.json"
        if not summary.exists():
            continue
        try:
            data = json.loads(summary.read_text())
        except Exception:
            continue
        f1 = data.get("f1_score")
        val_loss = data.get("val_loss")
        if f1 is None:
            continue
        try:
            f1f = float(f1)
        except Exception:
            continue
        try:
            vlf = float(val_loss) if val_loss is not None else 1e9
        except Exception:
            vlf = 1e9
        candidates.append((clf_dir.name, f1f, vlf))

    if not candidates:
        raise RuntimeError(f"No usable summary.json found in {run_dir}")

    candidates.sort(key=lambda x: (-x[1], x[2]))
    return candidates[0][0]


def predict_aacnn(model_path: Path, real_dir: Path, sim_dir: Path) -> pl.DataFrame:
    """
    Use AACnnClassifier.predict() to get a DataFrame(filename, prob_real, pred_class) over (real+sim).
    """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "backup" / "simulations-classifiers" / "src"))
    from classifiers.data.sources import FastaSource  # type: ignore
    from classifiers.data.data import Data  # type: ignore
    from classifiers.data import tokenizers  # type: ignore
    from classifiers.classif import AACnnClassifier  # type: ignore

    data = Data(source_real=FastaSource(real_dir), source_simulated=FastaSource(sim_dir), tokenizer=tokenizers.AA_TOKENIZER)
    clf = AACnnClassifier(
        data=data,
        out_path=model_path.parent,
        device="cpu",
        progress_bar=False,
        disable_compile=True,
    )
    df = clf.predict(model_path=model_path)
    return df


def resolve_disk_record(filename_key: str, real_dir: Path, sim_dir: Path) -> tuple[str, Path | None, int | None, str]:
    """
    Map a model filename key (may include _sim suffix) to:
    - disk_filename (without _sim suffix)
    - disk_path (real_dir/disk_filename or sim_dir/disk_filename)
    - true_label (REAL=1, SIM=0) if resolvable
    - label_text ("Réel"/"Simulé"/"Inconnu")
    """
    # Normalize: remove "_sim" suffix from stem (before extension)
    disk_filename = filename_key
    if filename_key.endswith(".fasta"):
        stem = filename_key[:-6]
        if stem.endswith("_sim"):
            disk_filename = stem[:-4] + ".fasta"
    elif filename_key.endswith(".fa"):
        stem = filename_key[:-3]
        if stem.endswith("_sim"):
            disk_filename = stem[:-4] + ".fa"

    # Prefer explicit suffix: *_sim -> simulated
    if filename_key.endswith("_sim.fasta") or filename_key.endswith("_sim.fa") or filename_key.endswith("_sim"):
        p = sim_dir / disk_filename
        if p.exists():
            return disk_filename, p, 0, "Simulé"
        return disk_filename, None, 0, "Simulé"

    # Otherwise: if exists in real, treat as real; else if exists in sim, treat as sim
    p_real = real_dir / disk_filename
    if p_real.exists():
        return disk_filename, p_real, 1, "Réel"
    p_sim = sim_dir / disk_filename
    if p_sim.exists():
        return disk_filename, p_sim, 0, "Simulé"
    return disk_filename, None, None, "Inconnu"


def main() -> None:
    run1_dir = CLASSIF_DIR / "run_1"
    run2_dir = CLASSIF_DIR / "run_2"
    run2_real = CLASSIF_DIR / "run_2_real"
    run2_sim = CLASSIF_DIR / "run_2_sim"

    if not run2_sim.exists() or not run2_real.exists():
        raise SystemExit("Missing run_2_real/ or run_2_sim/ — run two-iterations first.")

    log_path = find_latest_pipeline_log()
    if log_path is None:
        raise SystemExit("No pipeline_*.log found to read Youden thresholds.")

    thr1 = read_youden_threshold(1, log_path)
    thr2 = read_youden_threshold(2, log_path)
    if thr1 is None:
        raise SystemExit("Could not read RUN 1 Youden threshold from logs.")
    if thr2 is None:
        raise SystemExit("Could not read RUN 2 Youden threshold from logs.")

    best_clf_run1 = pick_best_classifier(run1_dir)
    best_clf_run2 = pick_best_classifier(run2_dir)

    if best_clf_run1 != "AACnnClassifier" or best_clf_run2 != "AACnnClassifier":
        raise SystemExit(
            f"Currently implemented predictor supports AACnnClassifier only. "
            f"Got best run1={best_clf_run1}, run2={best_clf_run2}."
        )

    model1 = run1_dir / best_clf_run1 / "best_model.pt"
    model2 = run2_dir / best_clf_run2 / "best_model.pt"
    if not model1.exists() or not model2.exists():
        raise SystemExit("Missing best_model.pt for run1 or run2 AACnnClassifier.")

    # Snapshot disk lists (used for existence checks)
    sim_files = sorted(run2_sim.glob("*.fasta"))
    real_files = sorted(run2_real.glob("*.fasta"))

    # Predict with both models on the Run2 dataset (real+sim)
    preds1 = predict_aacnn(model1, run2_real, run2_sim).rename({"prob_real": "prob_real_run1", "pred_class": "pred_class_run1"})
    preds2 = predict_aacnn(model2, run2_real, run2_sim).rename({"prob_real": "prob_real_run2", "pred_class": "pred_class_run2"})

    df = preds1.join(preds2, on="filename", how="inner")

    # Resolve disk identity (filename/path/md5/label) from model key
    filenames = df["filename"].to_list()
    disk_filenames: list[str] = []
    paths: list[str] = []
    md5s: list[str] = []
    true_labels: list[int | None] = []
    label_texts: list[str] = []

    for fn in filenames:
        disk_fn, p, tl, lt = resolve_disk_record(str(fn), run2_real, run2_sim)
        disk_filenames.append(disk_fn)
        label_texts.append(lt)
        true_labels.append(tl)
        if p is not None and p.exists():
            paths.append(str(p.resolve()))
            try:
                md5s.append(md5sum_file(p))
            except Exception:
                md5s.append("ERROR")
        else:
            paths.append("N/A")
            md5s.append("N/A")

    df = df.with_columns(
        pl.Series("disk_filename", disk_filenames),
        pl.Series("path", paths),
        pl.Series("md5", md5s),
        pl.Series("label_text", label_texts),
        pl.Series("true_label", true_labels),
    )

    # Passed run1/run2 based on thresholds (meaningful for SIMULATED only)
    df = df.with_columns(
        pl.lit(float(thr1)).alias("youden_run1"),
        pl.lit(float(thr2)).alias("youden_run2"),
        pl.when(pl.col("true_label") == 0)
        .then(pl.col("prob_real_run1") >= float(thr1))
        .otherwise(None)
        .alias("passed_run1"),
        pl.when(pl.col("true_label") == 0)
        .then(pl.col("prob_real_run2") >= float(thr2))
        .otherwise(None)
        .alias("passed_run2"),
    )

    # Convenience status for sims (how they behave vs thresholds)
    df = df.with_columns(
        pl.when(pl.col("true_label") == 1)
        .then(pl.lit("Réel"))
        .when((pl.col("passed_run1") == True) & (pl.col("passed_run2") == True))  # noqa: E712
        .then(pl.lit("Simulé (passé R1+R2)"))
        .when(pl.col("passed_run1") == True)  # noqa: E712
        .then(pl.lit("Simulé (passé R1)"))
        .when(pl.col("passed_run2") == True)  # noqa: E712
        .then(pl.lit("Simulé (passé R2)"))
        .otherwise(pl.lit("Simulé (filtré)"))
        .alias("status"),
    )

    out_dir = CLASSIF_DIR / "run_2"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "dashboard_dataset.parquet"
    df.write_parquet(out_path)

    print(f"✓ Wrote {out_path} ({df.shape[0]} rows).")
    print(f"  RUN1 Youden threshold: {thr1}")
    print(f"  RUN2 Youden threshold: {thr2}")


if __name__ == "__main__":
    main()


