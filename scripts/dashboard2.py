# scripts/dashboard2.py
# Dash dashboard complet avec callbacks et auto-refresh.
import io
import base64
import hashlib
from pathlib import Path
from urllib.parse import parse_qs, urlencode
import pandas as pd
import numpy as np
import dash
from dash import Dash, dcc, html, Output, Input, State, callback_context, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import flask
import traceback
import re

# Importer polars seulement si disponible
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Optional tree rendering (if available)
try:
    from Bio import Phylo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PHYLO = True
except Exception:
    HAS_PHYLO = False

# Using only matplotlib with Bio.Phylo for tree visualization

REFRESH_INTERVAL_MS = 600000  # auto-refresh every 10 minutes


COLOR_REAL = "#2E86AB"      # bleu
COLOR_SIM = "#A23B72"       # violet/rouge


def md5sum_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcule le MD5 d'un fichier (streaming, sans tout charger en RAM)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_md5_map(directory: Path) -> dict[str, str]:
    """
    Construit un mapping {filename -> md5} pour tous les .fasta d'un dossier.
    Retourne {} si le dossier n'existe pas.
    """
    if not directory.exists():
        return {}
    md5_map: dict[str, str] = {}
    for f in directory.glob("*.fasta"):
        try:
            md5_map[f.name] = md5sum_file(f)
        except Exception:
            # Ne pas casser le dashboard si un fichier est illisible
            md5_map[f.name] = "ERROR"
    return md5_map


def get_project_root():
    """Return project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def results_path():
    return get_project_root() / "results"


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_roc_data(classifier, run_number):
    """Charge les données ROC depuis les fichiers CSV ou calcule à partir de best_preds.parquet"""
    base_dir = results_path() / "classification"
    roc_file = base_dir / f"run_{run_number}" / "roc_data" / f"{classifier}_roc.csv"
    
    # D'abord essayer de charger depuis le CSV
    if roc_file.exists():
        return pd.read_csv(roc_file)
    
    # Sinon, essayer de calculer depuis best_preds.parquet
    preds_file = base_dir / f"run_{run_number}" / classifier / "best_preds.parquet"
    if not preds_file.exists():
        return None
    
    try:
        preds_df = pd.read_parquet(preds_file)
        
        # Identifier les colonnes nécessaires
        prob_col = None
        label_col = None
        
        if "prob_real" in preds_df.columns:
            prob_col = "prob_real"
        elif "prob" in preds_df.columns:
            prob_col = "prob"
        
        if "target" in preds_df.columns:
            label_col = "target"
        elif "true_label" in preds_df.columns:
            label_col = "true_label"
        
        if not prob_col or not label_col:
            return None
        
        # Calculer la courbe ROC
        from sklearn.metrics import roc_curve, auc
        y_true = preds_df[label_col].values
        y_score = preds_df[prob_col].values
        
        # Vérifier qu'on a les deux classes
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            return None
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        
        # Créer le DataFrame
        roc_df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds
        })
        
        return roc_df
    except Exception as e:
        print(f"Erreur lors du calcul de ROC pour {classifier} run_{run_number}: {e}")
        return None


def load_train_history(classifier, run_number):
    """Charge l'historique d'entraînement"""
    base_dir = results_path() / "classification"
    history_file = base_dir / f"run_{run_number}" / classifier / "train_history.parquet"
    
    if history_file.exists():
        return pd.read_parquet(history_file)
    return None


def get_best_classifier():
    """Détermine le meilleur classificateur basé sur le F1 score de run_1"""
    base_dir = results_path() / "classification"
    classifiers = ["AACnnClassifier", "DenseMsaClassifier", "DenseSiteClassifier", "LogisticRegressionClassifier"]
    
    best_clf = None
    best_f1 = -1
    
    for clf in classifiers:
        history = load_train_history(clf, 1)
        if history is not None:
            # Chercher f1 ou f1_score
            f1_col = "f1" if "f1" in history.columns else ("f1_score" if "f1_score" in history.columns else None)
            if f1_col:
                max_f1 = history[f1_col].max()
                if max_f1 > best_f1:
                    best_f1 = max_f1
                    best_clf = clf
    
    return best_clf if best_clf else "AACnnClassifier"


def get_alignment_length(fasta_path):
    """Lit un fichier FASTA et retourne la longueur de l'alignement"""
    try:
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
            sequences = []
            current_seq = ""
            for line in lines:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)
            
            if sequences:
                return len(sequences[0]), len(sequences)  # longueur alignement, nombre de séquences
    except:
        pass
    return None, None


def get_best_predictions_run2():
    """Charge et enrichit les meilleures prédictions du RUN 2 avec infos RUN 1 et longueurs"""
    base_dir = results_path() / "classification"
    dashboard_ds = base_dir / "run_2" / "dashboard_dataset.parquet"
    preds_run2_file = base_dir / "run_2" / "preds_run2.parquet"
    preds_run1_file = base_dir / "run_1" / "preds_run1.parquet"
    run2_sim_dir = base_dir / "run_2_sim"
    run2_real_dir = base_dir / "run_2_real"
    
    # Prefer the dedicated dashboard dataset if available
    if dashboard_ds.exists():
        try:
            if HAS_POLARS:
                df = pl.read_parquet(dashboard_ds).to_pandas()
            else:
                df = pd.read_parquet(dashboard_ds)
            # Keep only simulations that passed BOTH run1 and run2 thresholds
            if {"true_label", "passed_run1", "passed_run2"}.issubset(set(df.columns)):
                df = df[(df["true_label"] == 0) & (df["passed_run1"] == True) & (df["passed_run2"] == True)]  # noqa: E712
            # Sort by run2 score if available
            if "prob_real_run2" in df.columns:
                df = df.sort_values("prob_real_run2", ascending=False)
            return df
        except Exception as e:
            print(f"Erreur lors du chargement de dashboard_dataset.parquet: {e}")

    if not preds_run2_file.exists():
        return None
    
    try:
        # Charger les prédictions RUN 2
        if HAS_POLARS:
            df_run2 = pl.read_parquet(preds_run2_file)
            
            if "prob_real" not in df_run2.columns:
                return None
            
            # Si filename n'existe pas ou contient des valeurs incorrectes, essayer de le reconstruire
            need_rebuild = False
            if "filename" not in df_run2.columns:
                need_rebuild = True
            else:
                # Vérifier si les noms de fichiers sont tous identiques ou invalides (ex: tous "0")
                unique_filenames = df_run2["filename"].unique().to_list()
                if len(unique_filenames) == 1 and (unique_filenames[0] in ["0", "1", ""] or not str(unique_filenames[0]).endswith(".fasta")):
                    need_rebuild = True
            
            if need_rebuild:
                # Essayer de trouver les fichiers de simulation pour reconstruire les noms
                run2_sim_dir = base_dir / "run_2_sim"
                if run2_sim_dir.exists():
                    sim_files = sorted([f.name for f in run2_sim_dir.glob("*.fasta")])
                    if len(sim_files) == len(df_run2):
                        # Si le nombre correspond, utiliser les noms de fichiers
                        df_run2 = df_run2.with_columns([
                            pl.Series("filename", sim_files)
                        ])
                    else:
                        # Sinon, créer des noms génériques
                        df_run2 = df_run2.with_columns([
                            pl.Series("filename", [f"sim_{i}.fasta" for i in range(len(df_run2))])
                        ])
                else:
                    # Créer des noms génériques si le dossier n'existe pas
                    df_run2 = df_run2.with_columns([
                        pl.Series("filename", [f"sim_{i}.fasta" for i in range(len(df_run2))])
                    ])
            
            # Charger aussi les prédictions RUN 1 si elles existent
            df_run1 = None
            if preds_run1_file.exists():
                df_run1 = pl.read_parquet(preds_run1_file)
                # Renommer les colonnes pour éviter les conflits
                if "prob_real" in df_run1.columns:
                    df_run1 = df_run1.rename({"prob_real": "prob_real_run1"})
                if "pred_class" in df_run1.columns:
                    df_run1 = df_run1.rename({"pred_class": "pred_class_run1"})
            
            # Filtrer uniquement les simulations de RUN 2
            if not run2_sim_dir.exists():
                return None
            
            sim_files = {f.name for f in run2_sim_dir.glob("*.fasta")}
            df_filtered = df_run2.filter(pl.col("filename").is_in(list(sim_files)))
            
            # Fusionner avec RUN 1 si disponible
            if df_run1 is not None:
                df_filtered = df_filtered.join(
                    df_run1.select(["filename", "prob_real_run1", "pred_class_run1"]),
                    on="filename",
                    how="left"
                )
            
            # Convertir en pandas pour ajouter les longueurs
            df_pandas = df_filtered.to_pandas()
        else:
            # Fallback pandas
            df_run2 = pd.read_parquet(preds_run2_file)
            
            if "prob_real" not in df_run2.columns:
                return None
            
            # Si filename n'existe pas ou contient des valeurs incorrectes, essayer de le reconstruire
            need_rebuild = False
            if "filename" not in df_run2.columns:
                need_rebuild = True
            else:
                # Vérifier si les noms de fichiers sont tous identiques ou invalides (ex: tous "0")
                unique_filenames = df_run2["filename"].unique()
                if len(unique_filenames) == 1 and (unique_filenames[0] in ["0", "1", ""] or not str(unique_filenames[0]).endswith(".fasta")):
                    need_rebuild = True
            
            if need_rebuild:
                # Essayer de trouver les fichiers de simulation pour reconstruire les noms
                run2_sim_dir = base_dir / "run_2_sim"
                if run2_sim_dir.exists():
                    sim_files = sorted([f.name for f in run2_sim_dir.glob("*.fasta")])
                    if len(sim_files) == len(df_run2):
                        # Si le nombre correspond, utiliser les noms de fichiers
                        df_run2["filename"] = sim_files
                    else:
                        # Sinon, créer des noms génériques
                        df_run2["filename"] = [f"sim_{i}.fasta" for i in range(len(df_run2))]
                else:
                    # Créer des noms génériques si le dossier n'existe pas
                    df_run2["filename"] = [f"sim_{i}.fasta" for i in range(len(df_run2))]
            
            # Charger RUN 1
            df_run1 = None
            if preds_run1_file.exists():
                df_run1 = pd.read_parquet(preds_run1_file)
                df_run1 = df_run1.rename(columns={"prob_real": "prob_real_run1", "pred_class": "pred_class_run1"})
            
            # Filtrer simulations
            if not run2_sim_dir.exists():
                return None
            
            sim_files = {f.name for f in run2_sim_dir.glob("*.fasta")}
            df_pandas = df_run2[df_run2["filename"].isin(sim_files)]
            
            # Fusionner avec RUN 1
            if df_run1 is not None:
                df_pandas = df_pandas.merge(
                    df_run1[["filename", "prob_real_run1", "pred_class_run1"]],
                    on="filename",
                    how="left"
                )
        
        # Ajouter les longueurs d'alignement
        alignment_lengths = []
        num_sequences = []
        md5s = []
        paths = []
        md5_map_sim = build_md5_map(run2_sim_dir)
        md5_map_real = build_md5_map(run2_real_dir)
        
        for filename in df_pandas["filename"]:
            file_path = run2_sim_dir / filename
            length, n_seqs = get_alignment_length(file_path)
            alignment_lengths.append(length if length is not None else "N/A")
            num_sequences.append(n_seqs if n_seqs is not None else "N/A")
            # MD5: d'abord tenter dans run_2_sim, sinon run_2_real
            md5s.append(md5_map_sim.get(filename) or md5_map_real.get(filename) or "N/A")
            # Path: d'abord tenter run_2_sim, sinon run_2_real
            if (run2_sim_dir / filename).exists():
                paths.append(str((run2_sim_dir / filename).resolve()))
            elif (run2_real_dir / filename).exists():
                paths.append(str((run2_real_dir / filename).resolve()))
            else:
                paths.append("N/A")
        
        df_pandas["alignment_length"] = alignment_lengths
        df_pandas["num_sequences"] = num_sequences
        df_pandas["md5"] = md5s
        df_pandas["path"] = paths
        
        # Renommer les colonnes pour plus de clarté
        rename_dict = {
            "prob_real": "prob_real_run2",
            "pred_class": "pred_class_run2"
        }
        df_pandas = df_pandas.rename(columns=rename_dict)
        
        # Réorganiser les colonnes dans un ordre logique
        cols_order = ["filename"]
        
        # Ajouter les colonnes de RUN 1 si elles existent
        if "prob_real_run1" in df_pandas.columns:
            cols_order.append("prob_real_run1")
        if "pred_class_run1" in df_pandas.columns:
            cols_order.append("pred_class_run1")
        
        # Ajouter les colonnes de RUN 2
        if "prob_real_run2" in df_pandas.columns:
            cols_order.append("prob_real_run2")
        if "pred_class_run2" in df_pandas.columns:
            cols_order.append("pred_class_run2")
        
        # Ajouter les infos d'alignement
        cols_order.extend(["alignment_length", "num_sequences", "md5", "path"])
        
        # Ajouter les autres colonnes restantes
        for col in df_pandas.columns:
            if col not in cols_order:
                cols_order.append(col)
        
        df_pandas = df_pandas[cols_order]
        
        # Trier par prob_real_run2 décroissant
        df_sorted = df_pandas.sort_values("prob_real_run2", ascending=False)
        
        return df_sorted
        
    except Exception as e:
        print(f"Erreur lors du chargement des prédictions enrichies: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_run1_run2_scatter_plot(base_dir):
    """Crée un scatter plot comparant les scores Run1 vs Run2 avec couleurs pour réel/simulé
    et indique quelles séquences ont été gardées vs filtrées selon les seuils de Youden"""
    try:
        # Si un dataset dashboard "clean" existe, l'utiliser directement
        dashboard_ds = base_dir / "run_2" / "dashboard_dataset.parquet"
        if dashboard_ds.exists():
            try:
                if HAS_POLARS:
                    df = pl.read_parquet(dashboard_ds).to_pandas()
                else:
                    df = pd.read_parquet(dashboard_ds)

                # Scatter: comparer scores run1 vs run2, colorer par vrai label (réel/simulé),
                # afficher seuils Youden et passer l'échelle en % (0-100).
                required = {"prob_real_run1", "prob_real_run2"}
                if required.issubset(set(df.columns)):
                    # Scatter: montrer où se situent les RÉELS et les SIMULÉS.
                    # Le passage des seuils est encodé par les symboles (status) pour les simulés.

                    df["label_text"] = df.get("label_text", "Inconnu")
                    valid = {"Réel", "Simulé", "Inconnu"}
                    df["label_text"] = df["label_text"].apply(lambda x: x if x in valid else "Inconnu")

                    # Affichage "passé" lisible: utiliser status si présent
                    if "status" in df.columns:
                        color_col = "label_text"  # couleur simple Réel vs Simulé
                        symbol_col = "status"     # symbole encode passé R1/R2
                    else:
                        color_col = "label_text"
                        symbol_col = None

                    # Convertir en pourcentage pour l'affichage
                    df["x_pct"] = df["prob_real_run1"] * 100.0
                    df["y_pct"] = df["prob_real_run2"] * 100.0

                    # Symbol map (si status existe)
                    symbol_map = None
                    if symbol_col == "status":
                        symbol_map = {
                            "Réel": "circle",
                            "Simulé (passé R1+R2)": "diamond",
                            "Simulé (passé R1)": "triangle-up",
                            "Simulé (passé R2)": "triangle-down",
                            "Simulé (filtré)": "x",
                        }

                    fig = px.scatter(
                        df,
                        x="x_pct",
                        y="y_pct",
                        color=color_col,
                        symbol=symbol_col,
                        symbol_map=symbol_map,
                        color_discrete_map={"Réel": COLOR_REAL, "Simulé": COLOR_SIM, "Inconnu": "gray"},
                        hover_data=[c for c in ["filename", "md5", "path", "passed_run1", "passed_run2", "youden_run1", "youden_run2"] if c in df.columns],
                        labels={
                            "x_pct": "Score Run1 (%)",
                            "y_pct": "Score Run2 (%)",
                            "label_text": "Type",
                            "status": "Passage seuils",
                        },
                        title="Comparaison des scores Run1 vs Run2 (Meilleur classifieur) — Réel vs Simulé | lignes: Youden",
                    )

                    # Axes 0-100%
                    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Score Run1 (%)")
                    fig.update_yaxes(range=[0, 100], ticksuffix="%", title="Score Run2 (%)")

                    # Lignes de seuil Youden (si disponibles)
                    if "youden_run1" in df.columns and df["youden_run1"].notna().any():
                        y1 = float(df["youden_run1"].dropna().iloc[0]) * 100.0
                        fig.add_vline(x=y1, line_dash="dot", line_color="orange", annotation_text=f"Youden R1: {y1:.1f}%", annotation_position="top")
                    if "youden_run2" in df.columns and df["youden_run2"].notna().any():
                        y2 = float(df["youden_run2"].dropna().iloc[0]) * 100.0
                        fig.add_hline(y=y2, line_dash="dot", line_color="purple", annotation_text=f"Youden R2: {y2:.1f}%", annotation_position="right")

                    fig.update_layout(template="plotly_white", height=650, hovermode="closest")
                    return fig
            except Exception as e:
                print(f"Erreur lors du chargement de dashboard_dataset.parquet pour scatter: {e}")

        # Charger les prédictions Run1 et Run2
        preds_run1_file = base_dir / "run_1" / "preds_run1.parquet"
        preds_run2_file = base_dir / "run_2" / "preds_run2.parquet"
        
        if not preds_run1_file.exists() or not preds_run2_file.exists():
            return None
        
        # Récupérer les seuils de Youden depuis les logs
        run_stats = get_run_statistics()
        threshold_run1 = None
        threshold_run2 = None
        
        try:
            if run_stats.get('run1_threshold') != "N/A":
                threshold_run1 = float(run_stats['run1_threshold'])
            if run_stats.get('run2_threshold') != "N/A":
                threshold_run2 = float(run_stats['run2_threshold'])
        except (ValueError, TypeError):
            pass
        
        # Charger les données
        if HAS_POLARS:
            df_run1 = pl.read_parquet(preds_run1_file)
            df_run2 = pl.read_parquet(preds_run2_file)
            
            # Convertir en pandas pour Plotly
            df_run1_pd = df_run1.to_pandas()
            df_run2_pd = df_run2.to_pandas()
        else:
            df_run1_pd = pd.read_parquet(preds_run1_file)
            df_run2_pd = pd.read_parquet(preds_run2_file)
        
        # Identifier les colonnes de probabilité
        prob_col_run1 = "prob_real" if "prob_real" in df_run1_pd.columns else None
        prob_col_run2 = "prob_real" if "prob_real" in df_run2_pd.columns else None
        
        if not prob_col_run1 or not prob_col_run2:
            return None
        
        # Identifier les colonnes de label (réel/simulé)
        label_col = None
        for col in ["target", "true_label", "label"]:
            if col in df_run1_pd.columns:
                label_col = col
                break
        
        # Filtrer pour ne garder que les alignements qui ont été conservés après run1
        # (c'est-à-dire ceux qui sont présents dans run2)
        run2_real_dir = base_dir / "run_2_real"
        run2_sim_dir = base_dir / "run_2_sim"
        
        # Récupérer la liste des fichiers qui ont été gardés après run1
        kept_after_run1 = set()
        
        if run2_real_dir.exists():
            kept_after_run1.update(f.name for f in run2_real_dir.glob("*.fasta"))
        if run2_sim_dir.exists():
            kept_after_run1.update(f.name for f in run2_sim_dir.glob("*.fasta"))
        
        # Préparer MD5 (pour vérifier qu'on compare bien les mêmes fichiers)
        md5_map_sim = build_md5_map(run2_sim_dir)
        md5_map_real = build_md5_map(run2_real_dir)

        # Filtrer les DataFrames pour ne garder que les alignements conservés après run1
        if kept_after_run1:
            df_run1_pd_filtered = df_run1_pd[df_run1_pd["filename"].isin(kept_after_run1)].copy()
            df_run2_pd_filtered = df_run2_pd[df_run2_pd["filename"].isin(kept_after_run1)].copy()
        else:
            # Si les dossiers n'existent pas, utiliser tous les alignements (fallback)
            df_run1_pd_filtered = df_run1_pd.copy()
            df_run2_pd_filtered = df_run2_pd.copy()
        
        # Fusionner les données sur filename (seulement pour les alignements conservés)
        df_merged = df_run1_pd_filtered[["filename", prob_col_run1]].merge(
            df_run2_pd_filtered[["filename", prob_col_run2]],
            on="filename",
            how="inner",
            suffixes=("_run1", "_run2")
        )

        # Ajouter le MD5 au merged (utile pour débugger les "mêmes" fichiers)
        df_merged["md5"] = df_merged["filename"].map(
            lambda fn: md5_map_sim.get(fn) or md5_map_real.get(fn) or "N/A"
        )
        # Ajouter le path au merged (run_2_sim prioritaire, sinon run_2_real)
        def _resolve_path(fn: str) -> str:
            p_sim = run2_sim_dir / fn
            if p_sim.exists():
                return str(p_sim.resolve())
            p_real = run2_real_dir / fn
            if p_real.exists():
                return str(p_real.resolve())
            return "N/A"

        df_merged["path"] = df_merged["filename"].map(_resolve_path)
        
        # Ajouter les labels si disponibles
        if label_col and label_col in df_run1_pd.columns:
            labels = df_run1_pd[["filename", label_col]].set_index("filename")[label_col].to_dict()
            df_merged["label"] = df_merged["filename"].map(labels)
            # Mapper les labels en texte, en gérant les valeurs NaN/None
            df_merged["label_text"] = df_merged["label"].map({0: "Simulé", 1: "Réel"}).fillna("Inconnu")
        else:
            df_merged["label_text"] = "Inconnu"
        
        # S'assurer que toutes les valeurs sont bien dans le color_discrete_map
        # Remplacer toute valeur non reconnue par "Inconnu"
        valid_labels = {"Réel", "Simulé", "Inconnu"}
        df_merged["label_text"] = df_merged["label_text"].apply(
            lambda x: x if x in valid_labels else "Inconnu"
        )
        
        # Ajouter une colonne pour indiquer si la séquence simulée a été gardée ou filtrée
        # Une séquence simulée est gardée si prob_real >= seuil de Youden
        df_merged["kept_run1"] = False
        df_merged["kept_run2"] = False
        
        if threshold_run1 is not None:
            # Pour run1 : séquences simulées avec prob_real >= seuil
            mask_sim_run1 = df_merged["label_text"] == "Simulé"
            df_merged.loc[mask_sim_run1, "kept_run1"] = (
                df_merged.loc[mask_sim_run1, f"{prob_col_run1}_run1"] >= threshold_run1
            )
        
        if threshold_run2 is not None:
            # Pour run2 : séquences simulées avec prob_real >= seuil
            mask_sim_run2 = df_merged["label_text"] == "Simulé"
            df_merged.loc[mask_sim_run2, "kept_run2"] = (
                df_merged.loc[mask_sim_run2, f"{prob_col_run2}_run2"] >= threshold_run2
            )
        
        # Créer une colonne combinée pour l'affichage
        def get_status(row):
            if row["label_text"] == "Réel":
                return "Réel"
            elif row["label_text"] == "Simulé":
                kept1 = row.get("kept_run1", False)
                kept2 = row.get("kept_run2", False)
                if kept1 and kept2:
                    return "Simulé (gardée R1+R2)"
                elif kept1:
                    return "Simulé (gardée R1)"
                elif kept2:
                    return "Simulé (gardée R2)"
                else:
                    return "Simulé (filtrée)"
            else:
                return "Inconnu"
        
        df_merged["status"] = df_merged.apply(get_status, axis=1)
        
        # Vérifier qu'on a des données valides
        if df_merged.empty:
            return None
        
        # Créer le scatter plot avec les statuts détaillés
        use_go_scatter = False
        try:
            # Utiliser la colonne status pour un meilleur affichage
            color_map = {
                "Réel": "blue",
                "Simulé (gardée R1+R2)": "green",
                "Simulé (gardée R1)": "orange",
                "Simulé (gardée R2)": "yellow",
                "Simulé (filtrée)": "red",
                "Inconnu": "gray"
            }
            
            fig = px.scatter(
                df_merged,
                x=f"{prob_col_run1}_run1",
                y=f"{prob_col_run2}_run2",
                color="status",
                color_discrete_map=color_map,
                hover_data=["filename", "md5", "path"],
                labels={
                    f"{prob_col_run1}_run1": "Score Run1 (prob_real)",
                    f"{prob_col_run2}_run2": "Score Run2 (prob_real)",
                    "status": "Statut"
                },
                title="Comparaison des scores Run1 vs Run2 (alignements conservés après Run1)<br><sub>Lignes: seuils de Youden | Couleurs: statut de filtrage Run2</sub>"
            )
        except Exception as e:
            # Si px.scatter échoue, essayer avec go.Scatter directement
            print(f"Erreur avec px.scatter, utilisation de go.Scatter: {e}")
            use_go_scatter = True
            fig = go.Figure()
            
            # Ajouter les traces pour chaque statut
            color_map = {
                "Réel": "blue",
                "Simulé (gardée R1+R2)": "green",
                "Simulé (gardée R1)": "orange",
                "Simulé (gardée R2)": "yellow",
                "Simulé (filtrée)": "red",
                "Inconnu": "gray"
            }
            
            for status_type, color in color_map.items():
                mask = df_merged["status"] == status_type
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=df_merged.loc[mask, f"{prob_col_run1}_run1"],
                        y=df_merged.loc[mask, f"{prob_col_run2}_run2"],
                        mode='markers',
                        name=status_type,
                        marker=dict(color=color, size=8),
                        text=df_merged.loc[mask, "filename"],
                        customdata=df_merged.loc[mask, ["md5", "path"]].values,
                        hovertemplate='%{text}<br>MD5: %{customdata[0]}<br>Path: %{customdata[1]}<br>Run1: %{x:.3f}<br>Run2: %{y:.3f}<br>Statut: '
                        + status_type + '<extra></extra>'
                    ))
        
        # Ajouter les lignes de seuil de Youden
        max_val = max(df_merged[f"{prob_col_run1}_run1"].max(), df_merged[f"{prob_col_run2}_run2"].max())
        min_val = min(df_merged[f"{prob_col_run1}_run1"].min(), df_merged[f"{prob_col_run2}_run2"].min())
        
        # Ligne diagonale (y=x)
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='y=x',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        ))
        
        # Ligne verticale pour le seuil Run1
        if threshold_run1 is not None:
            fig.add_vline(
                x=threshold_run1,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Youden R1: {threshold_run1:.3f}",
                annotation_position="top",
                opacity=0.7
            )
        
        # Ligne horizontale pour le seuil Run2
        if threshold_run2 is not None:
            fig.add_hline(
                y=threshold_run2,
                line_dash="dot",
                line_color="purple",
                annotation_text=f"Youden R2: {threshold_run2:.3f}",
                annotation_position="right",
                opacity=0.7
            )
        
        # Mettre à jour le layout une seule fois à la fin
        layout_updates = {
            "template": "plotly_white",
            "height": 600,
            "hovermode": 'closest'
        }
        
        # Si on a utilisé go.Scatter (fallback), ajouter les titres d'axes
        if use_go_scatter:
            layout_updates.update({
                "xaxis_title": f"Score Run1 ({prob_col_run1})",
                "yaxis_title": f"Score Run2 ({prob_col_run2})",
                "title": "Comparaison des scores Run1 vs Run2 (alignements conservés après Run1)<br><sub>Lignes: seuils de Youden | Couleurs: statut de filtrage Run2</sub>"
            })
        
        fig.update_layout(**layout_updates)
        
        return fig
    
    except Exception as e:
        print(f"Erreur lors de la création du scatter plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_new_simulation_stats(base_dir):
    """Récupère les statistiques détaillées sur les nouvelles simulations générées dans Run2"""
    try:
        # Charger les prédictions des nouvelles simulations
        new_sim_preds_file = base_dir / "run_2" / "new_sim_predictions.parquet"
        
        if not new_sim_preds_file.exists():
            return None
        
        # Charger les données
        if HAS_POLARS:
            df = pl.read_parquet(new_sim_preds_file)
            df_pd = df.to_pandas()
        else:
            df_pd = pd.read_parquet(new_sim_preds_file)
        
        # Compter le nombre total de nouvelles simulations générées
        new_sim_dir = base_dir / "run_2" / "new_sim"
        num_generated = len(list(new_sim_dir.glob("*.fasta"))) if new_sim_dir.exists() else 0
        
        # Identifier la colonne de probabilité
        prob_col = "prob_real" if "prob_real" in df_pd.columns else None
        if not prob_col or df_pd.empty:
            return None
        
        scores = df_pd[prob_col].values
        
        # Calculer les statistiques de base
        num_selected = len(df_pd[df_pd[prob_col] >= 0.5])  # Au-dessus du seuil
        selection_rate = (num_selected / num_generated * 100) if num_generated > 0 else 0.0
        
        # Statistiques descriptives
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Quartiles
        q25 = np.percentile(scores, 25)
        q75 = np.percentile(scores, 75)
        
        # Pourcentages de sélection à différents seuils
        threshold_05 = len(df_pd[df_pd[prob_col] >= 0.5])
        threshold_07 = len(df_pd[df_pd[prob_col] >= 0.7])
        threshold_09 = len(df_pd[df_pd[prob_col] >= 0.9])
        
        return {
            "num_new_sims": num_generated,
            "num_selected": num_selected,
            "selection_rate": selection_rate,
            "median_score": median_score,
            "mean_score": mean_score,
            "std_score": std_score,
            "min_score": min_score,
            "max_score": max_score,
            "q25": q25,
            "q75": q75,
            "threshold_05": threshold_05,
            "threshold_07": threshold_07,
            "threshold_09": threshold_09,
            "scores": scores  # Pour les visualisations
        }
    
    except Exception as e:
        print(f"Erreur lors du calcul des statistiques des nouvelles simulations: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_run_statistics():
    """Récupère les statistiques sur les runs (données initiales, gardées après run 1 et run 2)"""
    base_dir = results_path() / "classification"
    
    stats = {
        "initial_real": 0,
        "initial_sim": 0,
        "run1_kept_sim": 0,
        "run1_threshold": "N/A",
        "run1_kept_percentage": 0.0,
        "run2_kept_sim": 0,
        "run2_threshold": "N/A",
        "run2_kept_percentage_from_run1": 0.0,
        "run2_kept_percentage_from_initial": 0.0,
        "run2_real_path": "N/A",
        "run2_sim_path": "N/A"
    }
    
    # Compter les données initiales
    initial_real_dir = Path("results/preprocessed/clean_data")
    initial_sim_dir = Path("results/simulations")
    
    if initial_real_dir.exists():
        stats["initial_real"] = len(list(initial_real_dir.glob("*.fasta")))
    if initial_sim_dir.exists():
        stats["initial_sim"] = len(list(initial_sim_dir.glob("*.fasta")))
    
    # Données gardées après run 1 (= données dans run_2)
    run2_real_dir = base_dir / "run_2_real"
    run2_sim_dir = base_dir / "run_2_sim"
    
    if run2_real_dir.exists():
        stats["run2_real_path"] = str(run2_real_dir)
    if run2_sim_dir.exists():
        stats["run2_sim_path"] = str(run2_sim_dir)
        stats["run1_kept_sim"] = len(list(run2_sim_dir.glob("*.fasta")))
    
    # Calculer les pourcentages pour run 1
    if stats["initial_sim"] > 0:
        stats["run1_kept_percentage"] = (stats["run1_kept_sim"] / stats["initial_sim"]) * 100
    
    # Chercher le threshold utilisé dans les logs (prendre le plus récent)
    log_files = list(base_dir.glob("pipeline_*.log"))
    log_path = None
    if log_files:
        log_path = max(log_files, key=lambda p: p.stat().st_mtime)

    if log_path:
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                # Chercher le pattern pour le threshold
                match = re.search(r"\[RUN 1\].*?optimal threshold = ([\d.]+)", content)
                if not match:
                    match = re.search(r"optimal threshold = ([\d.]+)", content)
                if match:
                    stats["run1_threshold"] = match.group(1)
        except:
            pass
    
    # Données après run 2 : extraire depuis les logs (formats multiples)
    if log_path:
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                # Chercher le pattern pour RUN 2
                # Ex: "[RUN 2] Selected 68 sims flagged REAL"
                # ou  "[RUN 2] Selected 68 sims flagged REAL (7.90% of initial 861 simulated alignments)"
                match = re.search(r"\[RUN 2\]\s+Selected\s+(\d+)\s+sims\s+flagged\s+REAL", content)
                if match:
                    stats["run2_kept_sim"] = int(match.group(1))
                    
                    # Calculer les pourcentages pour run 2
                    if stats["run1_kept_sim"] > 0:
                        stats["run2_kept_percentage_from_run1"] = (stats["run2_kept_sim"] / stats["run1_kept_sim"]) * 100
                    if stats["initial_sim"] > 0:
                        stats["run2_kept_percentage_from_initial"] = (stats["run2_kept_sim"] / stats["initial_sim"]) * 100
                
                # Aussi chercher le threshold de RUN 2
                match2 = re.search(r"\[RUN 2\].*?optimal threshold = ([\d.]+)", content)
                if match2:
                    stats["run2_threshold"] = match2.group(1)
        except Exception as e:
            print(f"Erreur lors de la lecture du log pour RUN 2: {e}")
    
    # Fallback: si le log n'a pas l'info RUN2, utiliser le dataset dashboard (source plus fiable)
    try:
        ds = load_dashboard_dataset_df(base_dir)
        if ds is not None and not ds.empty:
            ds_stats = compute_dashboard_stats(ds)
            if stats["run2_threshold"] == "N/A" and ds_stats.get("youden_run2") is not None:
                stats["run2_threshold"] = f"{ds_stats['youden_run2']:.4f}"
            if stats["run2_kept_sim"] == 0 and ds_stats.get("n_sim_passed_both") is not None:
                # Attention: ds_stats['n_sim_passed_both'] = simulés passés R1&R2 (plus strict),
                # mais c'est au moins une valeur non nulle si RUN2 existe.
                stats["run2_kept_sim"] = int(ds_stats["n_sim_passed_both"])
                if stats["run1_kept_sim"] > 0:
                    stats["run2_kept_percentage_from_run1"] = (stats["run2_kept_sim"] / stats["run1_kept_sim"]) * 100
                if stats["initial_sim"] > 0:
                    stats["run2_kept_percentage_from_initial"] = (stats["run2_kept_sim"] / stats["initial_sim"]) * 100
    except Exception:
        pass

    return stats


def load_dashboard_dataset_df(classif_dir: Path) -> pd.DataFrame | None:
    """Charge le dataset dashboard (Run2) si présent."""
    p = classif_dir / "run_2" / "dashboard_dataset.parquet"
    if not p.exists():
        return None
    try:
        if HAS_POLARS:
            return pl.read_parquet(p).to_pandas()
        return pd.read_parquet(p)
    except Exception as e:
        print(f"Erreur lors du chargement de {p}: {e}")
        return None


def compute_dashboard_stats(df: pd.DataFrame) -> dict:
    """Stats fiables basées sur dashboard_dataset.parquet."""
    stats: dict = {}
    stats["n_total"] = len(df)
    if "true_label" in df.columns:
        stats["n_real"] = int((df["true_label"] == 1).sum())
        stats["n_sim"] = int((df["true_label"] == 0).sum())
    else:
        stats["n_real"] = None
        stats["n_sim"] = None

    # Youden thresholds (run1/run2)
    stats["youden_run1"] = float(df["youden_run1"].dropna().iloc[0]) if "youden_run1" in df.columns and df["youden_run1"].notna().any() else None
    stats["youden_run2"] = float(df["youden_run2"].dropna().iloc[0]) if "youden_run2" in df.columns and df["youden_run2"].notna().any() else None

    # Passed counts on sims
    if {"true_label", "passed_run1", "passed_run2"}.issubset(set(df.columns)):
        sims = df[df["true_label"] == 0]
        stats["n_sim_passed_both"] = int(((sims["passed_run1"] == True) & (sims["passed_run2"] == True)).sum())  # noqa: E712
    else:
        stats["n_sim_passed_both"] = None

    return stats


def make_violin_plot(df: pd.DataFrame) -> go.Figure:
    """
    Violin plots Réel vs Simulé pour RUN1 et RUN2 (meilleur classifieur),
    avec couleurs cohérentes.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("RUN 1", "RUN 2"), horizontal_spacing=0.12)

    if df.empty or "true_label" not in df.columns:
        fig.add_annotation(text="Aucune donnée", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template="plotly_white", height=450)
        return fig

    violin_opacity = 0.6
    points_opacity = 0.35

    for col, run_name, col_idx, youden_col in [
        ("prob_real_run1", "RUN 1", 1, "youden_run1"),
        ("prob_real_run2", "RUN 2", 2, "youden_run2"),
    ]:
        if col not in df.columns:
            continue
        # Downsample simulés pour matcher le nombre de réels (comparaison équitable)
        real_vals_full = df.loc[df["true_label"] == 1, col].dropna().to_numpy()
        sim_vals_full = df.loc[df["true_label"] == 0, col].dropna().to_numpy()
        n = min(len(real_vals_full), len(sim_vals_full))
        if n == 0:
            continue
        rng = np.random.default_rng(42)
        real_vals = real_vals_full if len(real_vals_full) == n else rng.choice(real_vals_full, size=n, replace=False)
        sim_vals = sim_vals_full if len(sim_vals_full) == n else rng.choice(sim_vals_full, size=n, replace=False)

        for label_val, name, color in [(0, "Simulé", COLOR_SIM), (1, "Réel", COLOR_REAL)]:
            vals = sim_vals if label_val == 0 else real_vals
            fig.add_trace(
                go.Violin(
                    y=vals * 100.0,
                    name=f"{name} (n={len(vals)})",
                    legendgroup=name,
                    showlegend=(col_idx == 1),
                    line_color=color,
                    fillcolor=color,
                    opacity=violin_opacity,
                    box_visible=True,
                    meanline_visible=False,
                    points="outliers",
                    marker=dict(color=color, opacity=points_opacity, size=4),
                ),
                row=1,
                col=col_idx,
            )

        # Seuil Youden du run
        if youden_col in df.columns and df[youden_col].notna().any():
            thr = float(df[youden_col].dropna().iloc[0]) * 100.0
            fig.add_hline(
                y=thr,
                line_dash="dash",
                line_color="red",
                line_width=2,
                # Eviter d'empiéter sur le titre du subplot
                annotation_text=f"Youden: {thr:.1f}%",
                annotation_position="bottom right",
                annotation_font=dict(size=10),
                annotation_yshift=-20,
                row=1,
                col=col_idx,
            )

    fig.update_yaxes(range=[0, 100], ticksuffix="%", title_text="Probabilité prédite (%)", row=1, col=1)
    fig.update_yaxes(range=[0, 100], ticksuffix="%", title_text="Probabilité prédite (%)", row=1, col=2)
    fig.update_layout(template="plotly_white", height=520, title="Distribution des probabilités (Meilleur classifieur) — RUN 1 vs RUN 2")
    return fig


def make_histograms(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("RUN 1", "RUN 2"), horizontal_spacing=0.12)
    if df.empty or "true_label" not in df.columns:
        fig.add_annotation(text="Données insuffisantes pour histogrammes", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template="plotly_white", height=450)
        return fig

    for col, col_idx, youden_col in [
        ("prob_real_run1", 1, "youden_run1"),
        ("prob_real_run2", 2, "youden_run2"),
    ]:
        if col not in df.columns:
            continue
        sim_full = df.loc[df["true_label"] == 0, col].dropna().to_numpy()
        real_full = df.loc[df["true_label"] == 1, col].dropna().to_numpy()
        n = min(len(real_full), len(sim_full))
        if n == 0:
            continue
        rng = np.random.default_rng(42)
        sim = (sim_full if len(sim_full) == n else rng.choice(sim_full, size=n, replace=False)) * 100.0
        real = (real_full if len(real_full) == n else rng.choice(real_full, size=n, replace=False)) * 100.0

        fig.add_trace(go.Histogram(x=real, nbinsx=40, name=f"Réel (n={len(real)})", marker_color=COLOR_REAL, opacity=0.6, showlegend=(col_idx == 1)), row=1, col=col_idx)
        fig.add_trace(go.Histogram(x=sim, nbinsx=40, name=f"Simulé (n={len(sim)})", marker_color=COLOR_SIM, opacity=0.6, showlegend=(col_idx == 1)), row=1, col=col_idx)

        if youden_col in df.columns and df[youden_col].notna().any():
            thr = float(df[youden_col].dropna().iloc[0]) * 100.0
            # Mettre l'annotation en bas pour éviter les titres
            fig.add_vline(
                x=thr,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Youden: {thr:.1f}%",
                annotation_position="bottom",
                annotation_font=dict(size=10),
                annotation_yshift=-20,
                row=1,
                col=col_idx,
            )

    fig.update_layout(template="plotly_white", height=520, barmode="overlay", title="Histogrammes des probabilités — RUN 1 vs RUN 2")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(title_text="Fréquence")
    return fig


def make_boxplot_comparison(df: pd.DataFrame) -> go.Figure:
    # 4 boxplots lisibles: (RUN1, RUN2) x (Réel, Simulé) sans chevauchement
    fig = make_subplots(rows=1, cols=2, subplot_titles=("RUN 1", "RUN 2"), horizontal_spacing=0.12)
    if df.empty or "true_label" not in df.columns:
        fig.add_annotation(text="Données insuffisantes pour boxplot", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(template="plotly_white", height=450)
        return fig

    for col_idx, (run_col, youden_col) in enumerate(
        [("prob_real_run1", "youden_run1"), ("prob_real_run2", "youden_run2")],
        start=1,
    ):
        if run_col not in df.columns:
            continue

        # Même stratégie: échantillonner pour comparer à effectifs égaux
        real_full = df.loc[df["true_label"] == 1, run_col].dropna().to_numpy()
        sim_full = df.loc[df["true_label"] == 0, run_col].dropna().to_numpy()
        n = min(len(real_full), len(sim_full))
        if n == 0:
            continue
        rng = np.random.default_rng(42)
        real_sample = real_full if len(real_full) == n else rng.choice(real_full, size=n, replace=False)
        sim_sample = sim_full if len(sim_full) == n else rng.choice(sim_full, size=n, replace=False)

        fig.add_trace(
            go.Box(
                y=sim_sample * 100.0,
                name=f"Simulé (n={n})",
                marker_color=COLOR_SIM,
                boxmean=False,
                showlegend=(col_idx == 1),
                legendgroup="Simulé",
            ),
            row=1,
            col=col_idx,
        )
        fig.add_trace(
            go.Box(
                y=real_sample * 100.0,
                name=f"Réel (n={n})",
                marker_color=COLOR_REAL,
                boxmean=False,
                showlegend=(col_idx == 1),
                legendgroup="Réel",
            ),
            row=1,
            col=col_idx,
        )

        # Youden en bas (évite les titres)
        if youden_col in df.columns and df[youden_col].notna().any():
            thr = float(df[youden_col].dropna().iloc[0]) * 100.0
            fig.add_hline(
                y=thr,
                line_dash="dot",
                line_color="red",
                line_width=2,
                annotation_text=f"Youden: {thr:.1f}%",
                annotation_position="bottom right",
                annotation_font=dict(size=10),
                annotation_yshift=-20,
                row=1,
                col=col_idx,
            )

    fig.update_layout(
        template="plotly_white",
        height=550,
        title="Boxplots (Réel vs Simulé) — RUN 1 vs RUN 2",
        yaxis_title="Probabilité prédite (%)",
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def create_roc_curves_plotly(classifier):
    """Crée les courbes ROC interactives avec Plotly pour un classificateur"""
    fig = go.Figure()
    
    colors = {"run_1": "blue", "run_2": "green"}
    found_data = False
    
    try:
        for run_num in [1, 2]:
            roc_data = load_roc_data(classifier, run_num)
            if roc_data is not None and not roc_data.empty:
                found_data = True
                # Calculer AUC
                from sklearn.metrics import auc
                auc_score = auc(roc_data["fpr"], roc_data["tpr"])
                
                fig.add_trace(go.Scatter(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    mode='lines',
                    name=f'RUN {run_num} (AUC = {auc_score:.3f})',
                    line=dict(color=colors[f"run_{run_num}"], width=2),
                ))

                # Seuil optimal de Youden (max TPR - FPR) + point sur la courbe
                if "threshold" in roc_data.columns:
                    try:
                        j = (roc_data["tpr"] - roc_data["fpr"]).to_numpy()
                        idx = int(np.nanargmax(j))
                        youden_thr = float(roc_data["threshold"].iloc[idx])
                        youden_fpr = float(roc_data["fpr"].iloc[idx])
                        youden_tpr = float(roc_data["tpr"].iloc[idx])

                        fig.add_trace(go.Scatter(
                            x=[youden_fpr],
                            y=[youden_tpr],
                            mode="markers",
                            name=f"RUN {run_num} Youden: {youden_thr:.3f}",
                            marker=dict(size=10, color=colors[f"run_{run_num}"], symbol="x"),
                            showlegend=True,
                            hovertemplate="Youden threshold: %{customdata:.3f}<br>FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>",
                            customdata=[youden_thr],
                        ))
                    except Exception:
                        pass
        
        if found_data:
            # Ligne de référence (classifieur aléatoire)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', width=1, dash='dash')
            ))
        
        fig.update_layout(
            title=f"Courbes ROC - {classifier}",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            template="plotly_white",
            hovermode='closest',
            legend=dict(x=0.6, y=0.1),
            height=500,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        if not found_data:
            # Définir les limites de l'axe même sans données
            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])
            fig.add_annotation(
                text=f"Aucune donnée ROC disponible pour {classifier}<br><br>Les ROC sont calculées à partir de best_preds.parquet<br>qui doivent contenir 'prob_real' et 'target'",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12, color="orange"),
                align="center"
            )
    except Exception as e:
        fig.add_annotation(
            text=f"Erreur: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color="red")
        )
    
    return fig


def create_learning_curves_plotly(classifier):
    """Crée les courbes d'apprentissage interactives avec Plotly pour un classificateur"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('RUN 1', 'RUN 2'),
        vertical_spacing=0.15
    )
    
    colors_train = {"run_1": "blue", "run_2": "darkblue"}
    colors_val = {"run_1": "red", "run_2": "darkred"}
    found_data = False
    
    try:
        for idx, run_num in enumerate([1, 2], start=1):
            history = load_train_history(classifier, run_num)
            if history is not None and not history.empty and "epoch" in history.columns:
                found_data = True
                
                # Identifier les colonnes de loss
                train_loss_col = None
                val_loss_col = None
                
                for col in history.columns:
                    if "loss" in col.lower() and "val" not in col.lower():
                        train_loss_col = col
                    if "val" in col.lower() and "loss" in col.lower():
                        val_loss_col = col
                
                # Grouper par epoch (moyenne si plusieurs runs)
                grouped = history.groupby("epoch").mean(numeric_only=True).reset_index()
                
                # Train loss
                if train_loss_col and train_loss_col in grouped.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=grouped["epoch"],
                            y=grouped[train_loss_col],
                            mode='lines+markers',
                            name=f'Train Loss',
                            line=dict(color=colors_train[f"run_{run_num}"], width=2),
                            legendgroup=f'run{run_num}',
                            showlegend=(idx == 1)
                        ),
                        row=idx, col=1
                    )
                
                # Validation loss
                if val_loss_col and val_loss_col in grouped.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=grouped["epoch"],
                            y=grouped[val_loss_col],
                            mode='lines+markers',
                            name=f'Val Loss',
                            line=dict(color=colors_val[f"run_{run_num}"], width=2),
                            legendgroup=f'run{run_num}',
                            showlegend=(idx == 1)
                        ),
                        row=idx, col=1
                    )
        
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        
        fig.update_layout(
            title=f"Courbes d'apprentissage - {classifier}",
            template="plotly_white",
            hovermode='x unified',
            height=600
        )
        
        if not found_data:
            fig.add_annotation(
                text=f"Aucune donnée d'entraînement disponible pour {classifier}<br>Vérifiez que le classificateur a été entraîné",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="orange"),
                align="center"
            )
    except Exception as e:
        fig.add_annotation(
            text=f"Erreur: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=12, color="red")
        )
    
    return fig


def list_tree_files():
    td = results_path() / "trees"
    if not td.exists():
        return []
    return sorted([p for p in td.glob("*") if p.is_file()])


def make_app():
    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    server = app.server

    app.layout = dbc.Container(
        [
            html.H2("ApprentiPhylo - Pipeline Dashboard", className="my-3 text-center"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Source / Controls"),
                                    dbc.CardBody(
                                        [
                                            html.Div("Base results dir:"),
                                            html.Pre(str(results_path())),
                                            dbc.Button("Refresh now", id="btn-refresh", color="primary", className="mb-2"),
                                            html.Div(id="last-action", style={"fontSize": "12px", "color": "gray"}),
                                            html.Hr(),
                                            html.Div("Auto-refresh interval (minutes):"),
                                            dcc.Input(id="input-interval", type="number", value=REFRESH_INTERVAL_MS//60000, step=1, min=1),
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Files"),
                                    dbc.CardBody(
                                        [
                                            html.Div([
                                                html.Strong("Metric file:"),
                                                html.Div(id="metrics-file", className="mt-2 small")
                                            ], className="mb-3"),
                                            html.Hr(),
                                            html.Div([
                                                html.Strong("Classification files:"),
                                                html.Div(id="classif-file", className="mt-2 small")
                                            ], className="mb-3"),
                                            html.Hr(),
                                            html.Div([
                                html.Strong("Tree files:"),
                                html.P("Cliquez sur un arbre pour le visualiser", className="text-muted small mb-2"),
                                                html.Div(id="tree-list")
                                            ]),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs-main",
                                value="tab-sim",
                                children=[
                                    dcc.Tab(label="Simulation (MPD)", value="tab-sim"),
                                    dcc.Tab(label="Classification", value="tab-classif"),
                                    dcc.Tab(label="Selected Tree", value="tab-tree"),
                                    dcc.Tab(label="Logs", value="tab-logs"),
                                ],
                            ),
                            html.Div(id="tab-content", className="mt-3"),
                        ],
                        width=9,
                    ),
                ]
            ),
            # Interval for auto-refresh
            dcc.Interval(id="interval-refresh", interval=REFRESH_INTERVAL_MS, n_intervals=0),
            # Location component to manage URL
            dcc.Location(id="url", refresh=False),
        ],
        fluid=True,
    )

    # -------------------------
    # Callbacks
    # -------------------------

    @app.callback(
        Output("interval-refresh", "interval"),
        Input("input-interval", "value"),
    )
    def update_interval(minutes):
        """Convertit les minutes en millisecondes pour l'auto-refresh"""
        try:
            minutes = int(minutes)
            # Minimum 1 minute (60000 ms)
            return max(60000, minutes * 60000)
        except Exception:
            return REFRESH_INTERVAL_MS

    @app.callback(
        Output("metrics-file", "children"),
        Output("classif-file", "children"),
        Output("tree-list", "children"),
        Input("interval-refresh", "n_intervals"),
        Input("btn-refresh", "n_clicks"),
    )
    def refresh_file_lists(n_intervals, n_clicks):
        """List available files in the left column; updates periodically or on click."""
        try:
            # Metrics
            metrics_file = results_path() / "metrics" / "phylo_metrics.csv"
            if metrics_file.exists():
                mf_node = dbc.Badge(
                    [html.I(className="bi bi-check-circle me-1"), "phylo_metrics.csv"],
                    color="success",
                    className="me-1"
                )
            else:
                mf_node = dbc.Badge(
                    [html.I(className="bi bi-x-circle me-1"), "Non trouvé"],
                    color="secondary"
                )

            # Classification (attempt common filenames)
            cd = results_path() / "classification"
            classif_candidates = []
            if cd.exists():
                for candidate in sorted(cd.glob("*.csv")):
                    classif_candidates.append(
                        html.Div([
                            html.I(className="bi bi-file-earmark-spreadsheet me-1"),
                            html.Span(candidate.name, className="small")
                        ], className="mb-1")
                    )
            if not classif_candidates:
                classif_candidates = [html.Div(
                    [html.I(className="bi bi-info-circle me-1"), "Aucun CSV de classification"],
                    className="text-muted small"
                )]

            # Trees - Affichage amélioré avec liens cliquables
            trees = list_tree_files()
            if trees:
                tree_nodes = []
                for p in trees:
                    tree_nodes.append(
                        dcc.Link(
                            href=f"?tab=tab-tree&tree={p.name}",
                            children=dbc.ListGroupItem([
                                html.Div([
                                    html.I(className="bi bi-file-earmark-text me-2", style={"color": "#6c757d"}),
                                    html.Span(p.name, style={"fontSize": "13px"}),
                                    html.I(className="bi bi-arrow-right-circle ms-auto", style={"color": "#0dcaf0", "fontSize": "16px"}),
                                ], className="d-flex align-items-center")
                            ], className="py-2 px-3 list-group-item-action", style={
                                "border": "none", 
                                "borderBottom": "1px solid #f0f0f0"
                            }),
                            style={"textDecoration": "none", "color": "inherit"}
                        )
                    )
                
                # Créer la liste avec overflow
                tree_display = html.Div(tree_nodes, style={"maxHeight": "400px", "overflowY": "auto"})
            else:
                tree_display = html.Div("No tree files found in results/trees", className="text-muted text-center py-3")

            return mf_node, classif_candidates, tree_display

        except Exception as e:
            tb = traceback.format_exc()
            error_msg = html.Div(f"Error: {str(e)}", className="text-danger small")
            return error_msg, error_msg, error_msg

    # Callbacks pour synchroniser l'URL avec l'état
    @app.callback(
        Output("tabs-main", "value"),
        Output("last-action", "children"),
        Input("url", "search"),
        prevent_initial_call=False,
    )
    def sync_tab_from_url(search):
        """Synchronise l'onglet actif et affiche l'arbre sélectionné depuis l'URL."""
        if search:
            params = parse_qs(search.lstrip('?'))
            tab = params.get('tab', ['tab-sim'])[0]
            
            # Si un arbre est sélectionné dans l'URL, afficher le message
            if 'tree' in params:
                tree_name = params['tree'][0]
                return tab, f"Arbre sélectionné: {tree_name}"
            
            return tab, ""
        return "tab-sim", ""  # Onglet par défaut
    
    @app.callback(
        Output("url", "search", allow_duplicate=True),
        Input("tabs-main", "value"),
        State("url", "search"),
        prevent_initial_call=True,
    )
    def sync_url_from_tab(tab, current_search):
        """Met à jour l'URL quand l'utilisateur change d'onglet manuellement."""
        # Parser les paramètres actuels
        params = {}
        if current_search:
            params = parse_qs(current_search.lstrip('?'))
            # Convertir les listes en valeurs simples
            params = {k: v[0] if isinstance(v, list) else v for k, v in params.items()}
        
        # Mettre à jour le paramètre tab
        params['tab'] = tab
        
        # Reconstruire l'URL
        return '?' + urlencode(params)
    
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs-main", "value"),
        Input("url", "search"),
        Input("interval-refresh", "n_intervals"),
    )
    def render_tab(tab, url_search, _):
        """Render the main tab content depending on which tab is active."""
        try:
            if tab == "tab-sim":
                metrics_file = results_path() / "metrics_results" / "mpd_results.csv"
                if metrics_file.exists():
                    df = safe_read_csv(metrics_file)
                    if df.empty:
                        return html.Div("Metrics file exists but could not be read or is empty.")

                    return html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H5("Options d'affichage", className="mb-3"),
                                        dbc.Switch(
                                            id="outlier-toggle",
                                            label="Masquer les outliers (IQR method)",
                                            value=False,
                                            className="mb-2"
                                        ),
                                        html.Small(
                                            "Les outliers sont détectés avec la méthode IQR (Interquartile Range). "
                                            "Les valeurs en dehors de Q1-1.5*IQR et Q3+1.5*IQR sont considérées comme outliers.",
                                            className="text-muted"
                                        )
                                    ])
                                ], className="mb-3")
                            ], width=12)
                        ]),
                        html.Div(id="sim-plots-container")
                    ])
                else:
                    return html.Div("No metrics file found at results/metrics_results/mpd_results.csv")


            elif tab == "tab-classif":
                classif_dir = results_path() / "classification"
                if not classif_dir.exists():
                    return html.Div("No classification directory found (results/classification).")

                predictions_dir = classif_dir / "predictions_analysis"
                
                # Déterminer le meilleur classificateur
                best_clf = get_best_classifier()
                
                # Récupérer les statistiques des runs
                run_stats = get_run_statistics()
                ds_df = load_dashboard_dataset_df(classif_dir)
                ds_stats = compute_dashboard_stats(ds_df) if ds_df is not None else None
                
                children = [
                    html.H3("Résultats de Classification", className="mb-4 text-center"),
                    html.Hr(),
                    
                    # Section Statistiques
                    html.H4("Statistiques des Runs", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Données initiales"),
                                dbc.CardBody([
                                    html.P([html.Strong("Réelles: "), f"{run_stats['initial_real']}"]),
                                    html.P([html.Strong("Simulées: "), f"{run_stats['initial_sim']}"]),
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Après RUN 1"),
                                dbc.CardBody([
                                    html.P([html.Strong("Simulées gardées: "), f"{run_stats['run1_kept_sim']}"]),
                                    html.P([html.Strong("Pourcentage: "), f"{run_stats['run1_kept_percentage']:.2f}%"]),
                                    html.P([html.Strong("Threshold: "), f"{run_stats['run1_threshold']}"]),
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Après RUN 2"),
                                dbc.CardBody([
                                    html.P([html.Strong("Simulées gardées (log): "), f"{run_stats['run2_kept_sim']}"]),
                                    html.P([html.Strong("% depuis RUN 1 (log): "), f"{run_stats['run2_kept_percentage_from_run1']:.2f}%"]),
                                    html.P([html.Strong("% depuis initial (log): "), f"{run_stats['run2_kept_percentage_from_initial']:.2f}%"]),
                                    html.P([html.Strong("Threshold RUN2 (log): "), f"{run_stats.get('run2_threshold', 'N/A')}"]),
                                    html.Hr(),
                                    html.P([html.Strong("Dataset dashboard: "), "OK" if ds_stats else "N/A"]),
                                    html.P([html.Strong("Youden RUN1 (dataset): "), f"{ds_stats['youden_run1']:.4f}" if ds_stats and ds_stats.get("youden_run1") is not None else "N/A"]),
                                    html.P([html.Strong("Youden RUN2 (dataset): "), f"{ds_stats['youden_run2']:.4f}" if ds_stats and ds_stats.get("youden_run2") is not None else "N/A"]),
                                    html.P([html.Strong("Simulés passés R1&R2 (dataset): "), f"{ds_stats['n_sim_passed_both']}" if ds_stats and ds_stats.get("n_sim_passed_both") is not None else "N/A"]),
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Chemins des données utilisées pour RUN 2"),
                                dbc.CardBody([
                                    html.P([html.Strong("Réelles: ")], style={"fontSize": "10px"}),
                                    html.P(run_stats['run2_real_path'], style={"fontSize": "9px", "wordBreak": "break-all"}),
                                    html.P([html.Strong("Simulées (gardées après RUN 1): ")], style={"fontSize": "10px"}),
                                    html.P(run_stats['run2_sim_path'], style={"fontSize": "9px", "wordBreak": "break-all"}),
                                    html.P("Note: Ces simulations ont été gardées après RUN 1 et réutilisées pour RUN 2", 
                                           style={"fontSize": "8px", "color": "gray", "marginTop": "5px"}),
                                ])
                            ])
                        ], width=3),
                    ], className="mb-4"),
                    html.Hr(),
                ]
                
                # ==========================================
                # SECTION 1-2: PLOTS DIRECTS (sans iframe) depuis dashboard_dataset.parquet
                # ==========================================
                if ds_df is not None and not ds_df.empty:
                    # Pour les distributions (violin/hist/box): afficher le dataset complet RUN2 (Réel + Simulé)
                    # afin d'avoir des histogrammes corrects (pas biaisés par le filtre "passé").
                    ds_view = ds_df

                    children.extend([
                        html.H4(f"Distribution des probabilités (Meilleur: {best_clf})", className="mb-3"),
                        html.P(
                            "Réel (bleu) vs Simulé (violet). Les seuils de Youden sont indiqués sur chaque run.",
                            className="text-muted",
                        ),
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=make_violin_plot(ds_view), id="violin-direct")], width=12),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([dcc.Graph(figure=make_histograms(ds_view), id="hist-direct")], width=6),
                            dbc.Col([dcc.Graph(figure=make_boxplot_comparison(ds_view), id="box-direct")], width=6),
                        ], className="mb-4"),
                        html.Hr(className="my-4"),
                    ])
                else:
                    children.append(html.Div("Dataset dashboard non trouvé. Lance: uv run python scripts/build_dashboard_dataset.py", className="alert alert-warning"))
                
                # ==========================================
                # SECTION 3: SÉLECTION DU CLASSIFICATEUR
                # ==========================================
                children.extend([
                    html.H4("Analyse détaillée par classificateur", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Sélectionner un classificateur:", className="fw-bold"),
                            dcc.Dropdown(
                                id="classifier-dropdown-tab",
                                options=[
                                    {"label": "AACnnClassifier", "value": "AACnnClassifier"},
                                    {"label": "DenseMsaClassifier", "value": "DenseMsaClassifier"},
                                    {"label": "DenseSiteClassifier", "value": "DenseSiteClassifier"},
                                    {"label": "LogisticRegressionClassifier", "value": "LogisticRegressionClassifier"},
                                ],
                                value=best_clf,
                                clearable=False,
                            ),
                        ], width=6)
                    ], className="mb-3"),
                    html.Div(id="classifier-details"),
                ])
                
                # ==========================================
                # SECTION 4: SCATTER PLOT RUN1 vs RUN2
                # ==========================================
                scatter_fig = create_run1_run2_scatter_plot(classif_dir)
                if scatter_fig:
                    children.extend([
                        html.Hr(className="my-4"),
                        html.H4("Comparaison des scores Run1 vs Run2", className="mb-3"),
                        html.P(
                            "Scatter plot montrant les scores de prédiction dans Run1 et Run2. "
                            "⚠️ Seuls les alignements conservés après Run1 sont affichés (présents dans run_2_real/ et run_2_sim/). "
                            "Les lignes indiquent les seuils de Youden. Les couleurs indiquent le statut de filtrage dans Run2: "
                            "Réel (bleu), Simulé gardée dans R1+R2 (vert), Simulé gardée seulement R1 (orange), "
                            "Simulé gardée seulement R2 (jaune), Simulé filtrée (rouge).",
                            className="text-muted"
                        ),
                        dcc.Graph(figure=scatter_fig, id="scatter-run1-run2"),
                        html.Hr(className="my-4"),
                    ])
                
                # ==========================================
                # SECTION 5: STATISTIQUES NOUVELLES SIMULATIONS
                # ==========================================
                new_sim_stats = get_new_simulation_stats(classif_dir)
                if new_sim_stats:
                    # Créer un histogramme des scores des nouvelles simulations
                    hist_fig = None
                    if "scores" in new_sim_stats and len(new_sim_stats["scores"]) > 0:
                        hist_fig = px.histogram(
                            x=new_sim_stats["scores"],
                            nbins=50,
                            title="Distribution des scores des nouvelles simulations",
                            labels={"x": "Probabilité d'être réel", "y": "Nombre de simulations"},
                            marginal="box"
                        )
                        hist_fig.update_layout(
                            template="plotly_white",
                            height=400,
                            showlegend=False
                        )
                        # Ajouter des lignes verticales pour les seuils
                        hist_fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                                          annotation_text="Seuil 0.5", annotation_position="top")
                        hist_fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                                          annotation_text="Seuil 0.7", annotation_position="top")
                        hist_fig.add_vline(x=new_sim_stats["median_score"], line_dash="dot", 
                                          line_color="blue", annotation_text="Médiane", annotation_position="top")
                    
                    children.extend([
                        html.H4("Statistiques détaillées sur les nouvelles simulations (Run2)", className="mb-3"),
                        html.P(
                            "Ces statistiques concernent les nouvelles simulations générées entre Run1 et Run2 "
                            "pour équilibrer le nombre de données simulées dans le dataset d'entraînement de Run2.",
                            className="text-muted mb-3"
                        ),
                        # Première ligne: Métriques principales
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Nombre généré"),
                                    dbc.CardBody([
                                        html.H3(f"{new_sim_stats['num_new_sims']}", className="text-center"),
                                        html.P("Nouvelles simulations", className="text-muted text-center mb-0")
                                    ])
                                ])
                            ], width=2),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Sélectionnées (≥0.5)"),
                                    dbc.CardBody([
                                        html.H3(f"{new_sim_stats['num_selected']}", className="text-center"),
                                        html.P(f"({new_sim_stats['selection_rate']:.1f}%)", className="text-muted text-center mb-0")
                                    ])
                                ])
                            ], width=2),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Score médian"),
                                    dbc.CardBody([
                                        html.H3(f"{new_sim_stats['median_score']:.3f}", className="text-center"),
                                        html.P("Probabilité médiane", className="text-muted text-center mb-0")
                                    ])
                                ])
                            ], width=2),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Score moyen"),
                                    dbc.CardBody([
                                        html.H3(f"{new_sim_stats['mean_score']:.3f}", className="text-center"),
                                        html.P("Probabilité moyenne", className="text-muted text-center mb-0")
                                    ])
                                ])
                            ], width=2),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Écart-type"),
                                    dbc.CardBody([
                                        html.H3(f"{new_sim_stats['std_score']:.3f}", className="text-center"),
                                        html.P("Dispersion", className="text-muted text-center mb-0")
                                    ])
                                ])
                            ], width=2),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Plage"),
                                    dbc.CardBody([
                                        html.H5(f"{new_sim_stats['min_score']:.3f} - {new_sim_stats['max_score']:.3f}", 
                                               className="text-center mb-0"),
                                        html.P("Min - Max", className="text-muted text-center mb-0")
                                    ])
                                ])
                            ], width=2),
                        ], className="mb-3"),
                        # Deuxième ligne: Quartiles et seuils
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Quartiles"),
                                    dbc.CardBody([
                                        html.P([html.Strong("Q25: "), f"{new_sim_stats['q25']:.3f}"], className="mb-1"),
                                        html.P([html.Strong("Q75: "), f"{new_sim_stats['q75']:.3f}"], className="mb-0"),
                                    ])
                                ])
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Sélection par seuil"),
                                    dbc.CardBody([
                                        html.P([html.Strong("≥ 0.5: "), f"{new_sim_stats['threshold_05']} simulations"], className="mb-1"),
                                        html.P([html.Strong("≥ 0.7: "), f"{new_sim_stats['threshold_07']} simulations"], className="mb-1"),
                                        html.P([html.Strong("≥ 0.9: "), f"{new_sim_stats['threshold_09']} simulations"], className="mb-0"),
                                    ])
                                ])
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Taux de sélection"),
                                    dbc.CardBody([
                                        html.P([html.Strong("Seuil 0.5: "), 
                                               f"{(new_sim_stats['threshold_05']/new_sim_stats['num_new_sims']*100):.1f}%"], 
                                              className="mb-1"),
                                        html.P([html.Strong("Seuil 0.7: "), 
                                               f"{(new_sim_stats['threshold_07']/new_sim_stats['num_new_sims']*100):.1f}%"], 
                                              className="mb-1"),
                                        html.P([html.Strong("Seuil 0.9: "), 
                                               f"{(new_sim_stats['threshold_09']/new_sim_stats['num_new_sims']*100):.1f}%"], 
                                              className="mb-0"),
                                    ])
                                ])
                            ], width=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("IQR"),
                                    dbc.CardBody([
                                        html.H5(f"{new_sim_stats['q75'] - new_sim_stats['q25']:.3f}", 
                                               className="text-center mb-0"),
                                        html.P("Interquartile Range", className="text-muted text-center mb-0")
                                    ])
                                ])
                            ], width=3),
                        ], className="mb-4"),
                        # Histogramme de distribution
                        html.Div([
                            dcc.Graph(figure=hist_fig, id="new-sim-histogram") if hist_fig else 
                            html.Div("Données insuffisantes pour générer l'histogramme", className="alert alert-info")
                        ], className="mb-4"),
                        html.Hr(className="my-4"),
                    ])
                
                # ==========================================
                # SECTION 6: MEILLEURES PRÉDICTIONS RUN 2
                # ==========================================
                best_preds_df = get_best_predictions_run2()
                if best_preds_df is not None and not best_preds_df.empty:
                    children.extend([
                        html.Hr(className="my-4"),
                        html.H4("Meilleures prédictions RUN 2 (Analyse complète)", className="mb-3"),
                        html.P([
                            f"Top {min(50, len(best_preds_df))} simulations classées par probabilité RUN 2. ",
                            html.Strong("Colonnes : "),
                            "filename, prob_real_run1 (RUN 1), pred_class_run1, prob_real_run2 (RUN 2), pred_class_run2, longueur alignement, nombre de séquences"
                        ], className="text-muted"),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    dbc.Table.from_dataframe(
                                        best_preds_df.head(50).round(4),  # Arrondir à 4 décimales
                                        striped=True,
                                        bordered=True,
                                        hover=True,
                                        responsive=True,
                                        size="sm"
                                    )
                                ], style={"maxHeight": "500px", "overflowY": "scroll"})
                            ])
                        ], className="mb-4"),
                        html.P([
                            html.Strong(f"Total: "),
                            f"{len(best_preds_df)} simulations dans le RUN 2 | ",
                            html.Strong("Légende: "),
                            "pred_class: 0=Simulé, 1=Réel"
                        ], className="text-muted")
                    ])
                else:
                    children.extend([
                        html.Hr(className="my-4"),
                        html.Div("Aucune prédiction RUN 2 disponible. Vérifiez que le RUN 2 a été exécuté.", className="alert alert-warning")
                    ])
                
                return html.Div(children)


            elif tab == "tab-tree":
                # Extraire le nom de l'arbre depuis l'URL
                tree_name = None
                if url_search:
                    params = parse_qs(url_search.lstrip('?'))
                    if 'tree' in params:
                        tree_name = params['tree'][0]
                
                # Debug: afficher les paramètres URL
                debug_info = dbc.Alert([
                    html.Strong("Debug - URL params:"),
                    html.Br(),
                    html.Small(f"URL search: {url_search}", className="text-muted"),
                    html.Br(),
                    html.Small(f"Tree name: {tree_name}", className="text-muted")
                ], color="light", className="mb-2")
                
                if not tree_name:
                    return html.Div([
                        debug_info,
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className="bi bi-info-circle", style={"fontSize": "48px", "color": "#6c757d"}),
                                    html.H4("Aucun arbre sélectionné", className="mt-3"),
                                    html.P("Cliquez sur un arbre dans la liste de gauche pour le visualiser.", 
                                           className="text-muted")
                                ], className="text-center py-5")
                            ])
                        ])
                    ])
                
                # Construire le chemin complet
                p = results_path() / "trees" / tree_name
                if not p.exists():
                    return html.Div([
                        debug_info,
                        dbc.Alert(f"Le fichier d'arbre '{tree_name}' n'existe plus. Chemin: {p}", color="danger")
                    ])
                
                # Lire le fichier Newick
                raw = p.read_text()
                
                content = [
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4([
                                html.I(className="bi bi-file-earmark-text me-2"),
                                f"Arbre: {p.name}"
                            ], className="mb-0")
                        ]),
                        dbc.CardBody([
                            # Informations sur l'arbre
                            dbc.Row([
                                dbc.Col([
                                    html.Strong("Nom du fichier: "),
                                    html.Span(p.name)
                                ], width=6),
                                dbc.Col([
                                    html.Strong("Taille: "),
                                    html.Span(f"{len(raw)} caractères")
                                ], width=6),
                            ], className="mb-3"),
                            
                            # Bouton de téléchargement
                            dbc.Button(
                                [html.I(className="bi bi-download me-2"), "Télécharger l'arbre"],
                                href=f"/download-tree/{p.name}",
                                color="primary",
                                size="sm",
                                external_link=True,
                                className="mb-3"
                            ),
                            
                            # Format Newick
                            html.H5("Format Newick:", className="mt-3"),
                            dbc.Card([
                                dbc.CardBody([
                                    html.Pre(
                                        raw[:2000] + ("..." if len(raw) > 2000 else ""),
                                        style={
                                            "backgroundColor": "#f8f9fa",
                                            "padding": "15px",
                                            "borderRadius": "5px",
                                            "fontSize": "12px",
                                            "overflowX": "auto",
                                            "maxHeight": "200px",
                                            "overflowY": "auto"
                                        }
                                    )
                                ])
                            ], className="mb-3"),
                        ])
                    ], className="mb-3")
                ]
                
                # Visualisation de l'arbre avec Bio.Phylo et matplotlib
                if HAS_PHYLO:
                    try:
                        tree = Phylo.read(str(p), "newick")
                        n_leaves = len(tree.get_terminals())
                        
                        # Style amélioré avec matplotlib
                        try:
                            plt.style.use('seaborn-v0_8-whitegrid')
                        except:
                            try:
                                plt.style.use('seaborn-whitegrid')
                            except:
                                plt.style.use('default')
                        base_height = 10
                        height_per_leaf = 0.4
                        fig_height = max(base_height, n_leaves * height_per_leaf)
                        fig_width = min(16, 12 + (n_leaves / 15))
                        
                        figfile = io.BytesIO()
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                        fig.patch.set_facecolor('white')
                        ax.set_facecolor('white')
                        
                        Phylo.draw(tree, axes=ax, do_show=False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_color('#2d3436')
                        ax.spines['left'].set_color('#2d3436')
                        ax.tick_params(colors='#2d3436')
                        ax.grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        plt.savefig(figfile, format="png", bbox_inches="tight", dpi=200, facecolor='white', edgecolor='none')
                        plt.close()
                        figfile.seek(0)
                        encoded = base64.b64encode(figfile.read()).decode("ascii")
                        
                        content.append(
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5([
                                        html.I(className="bi bi-diagram-3 me-2"),
                                        "Visualisation de l'arbre phylogénétique"
                                    ], className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dbc.Alert([
                                        html.Strong(f"Nombre de feuilles: {n_leaves}"),
                                        html.Br(),
                                        html.Small("Visualisation avec Bio.Phylo et matplotlib", className="text-muted")
                                    ], color="info", className="mb-3"),
                                    html.Img(
                                        src="data:image/png;base64," + encoded,
                                        style={
                                            "maxWidth": "100%", 
                                            "border": "2px solid #dee2e6", 
                                            "borderRadius": "8px",
                                            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                                        }
                                    )
                                ])
                            ])
                        )
                    except Exception as e:
                        content.append(
                            dbc.Alert([
                                html.Strong("Erreur de visualisation: "),
                                str(e)
                            ], color="warning")
                        )
                else:
                    content.append(
                        dbc.Alert([
                            html.Strong("Visualisation non disponible"),
                            html.Br(),
                            "Installez biopython et matplotlib pour activer la visualisation des arbres. "
                        ], color="info")
                    )
                
                return html.Div([debug_info] + content)

            elif tab == "tab-logs":
                lf = get_project_root() / "logs" / "pipeline_log.csv"
                if not lf.exists():
                    return dbc.Alert([
                        html.I(className="bi bi-exclamation-triangle me-2"),
                        "Aucun fichier de logs trouvé (logs/pipeline_log.csv)"
                    ], color="warning")
                
                try:
                    # Lire et parser le CSV
                    df_logs = pd.read_csv(lf)
                    
                    if df_logs.empty:
                        return dbc.Alert("Le fichier de logs est vide.", color="info")
                    
                    # Inverser l'ordre pour voir les logs les plus récents en premier
                    df_logs = df_logs.iloc[::-1]
                    
                    # Formater la durée en minutes et secondes si c'est en secondes
                    if "duration" in df_logs.columns:
                        def format_duration(seconds):
                            try:
                                s = float(seconds)
                                if s < 60:
                                    return f"{s:.1f}s"
                                else:
                                    minutes = int(s // 60)
                                    secs = int(s % 60)
                                    return f"{minutes}m {secs}s"
                            except:
                                return str(seconds)
                        
                        df_logs["duration"] = df_logs["duration"].apply(format_duration)
                    
                    # Colorer le statut
                    def colorize_status(row):
                        status = str(row["status"]).lower() if pd.notna(row["status"]) else ""
                        status_str = str(row["status"]) if pd.notna(row["status"]) else "N/A"
                        if "success" in status:
                            return "[OK] " + status_str
                        elif "error" in status:
                            return "[ERROR] " + status_str
                        else:
                            return "[INFO] " + status_str
                    
                    if "status" in df_logs.columns:
                        df_logs["status"] = df_logs.apply(colorize_status, axis=1)
                    
                    # Limiter à 100 dernières entrées
                    display_df = df_logs.head(100)
                    
                    return html.Div([
                        dbc.Card([
                            dbc.CardHeader([
                                html.H4([
                                    html.I(className="bi bi-clock-history me-2"),
                                    "Historique du Pipeline"
                                ], className="mb-0")
                            ]),
                            dbc.CardBody([
                                dbc.Alert([
                                    html.Strong(f"Total d'entrées: {len(df_logs)} | "),
                                    f"Affichage des {min(100, len(df_logs))} plus récentes"
                                ], color="info", className="mb-3"),
                                
                                html.Div([
                                    dbc.Table.from_dataframe(
                                        display_df,
                                        striped=True,
                                        bordered=True,
                                        hover=True,
                                        responsive=True,
                                        size="sm",
                                        style={"fontSize": "13px"}
                                    )
                                ], style={"maxHeight": "600px", "overflowY": "auto"}),
                                
                                html.Hr(),
                                dbc.Button(
                                    [html.I(className="bi bi-download me-2"), "Télécharger les logs complets"],
                                    color="secondary",
                                    size="sm",
                                    href=str(lf),
                                    external_link=True
                                )
                            ])
                        ])
                    ])
                    
                except Exception as e:
                    return dbc.Alert([
                        html.Strong("Erreur lors du chargement des logs: "),
                        str(e)
                    ], color="danger")

            else:
                return html.Div("Tab not implemented.")
        except Exception as e:
            return html.Div("Error rendering tab: " + str(e) + "\n" + traceback.format_exc())

    # Callback pour gérer l'affichage des plots avec/sans outliers
    @app.callback(
        Output("sim-plots-container", "children"),
        Input("outlier-toggle", "value"),
        Input("interval-refresh", "n_intervals"),
    )
    def update_sim_plots(remove_outliers, _):
        """Met à jour les plots de simulation avec ou sans outliers"""
        metrics_file = results_path() / "metrics_results" / "mpd_results.csv"
        if not metrics_file.exists():
            return html.Div("No metrics file found at results/metrics_results/mpd_results.csv")
        
        df = safe_read_csv(metrics_file)
        if df.empty:
            return html.Div("Metrics file exists but could not be read or is empty.")
        
        # Filtrer les outliers si demandé (méthode IQR)
        if remove_outliers and "MPD" in df.columns:
            total_count = len(df)
            Q1 = df["MPD"].quantile(0.25)
            Q3 = df["MPD"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_filtered = df[(df["MPD"] >= lower_bound) & (df["MPD"] <= upper_bound)]
            num_outliers = total_count - len(df_filtered)
            df = df_filtered
            
            outlier_info = dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                f"Outliers masqués: {num_outliers} points ({num_outliers/total_count*100:.1f}%) "
                f"en dehors de [{lower_bound:.3f}, {upper_bound:.3f}]"
            ], color="info", className="mb-3")
        else:
            outlier_info = html.Div()
        
        # Histogramme MPD
        fig_hist = px.histogram(
            df, x="MPD", nbins=40, title="MPD – Histogramme",
            marginal="box",
            template="plotly_white"
        )
        fig_hist.update_layout(height=500)

        # Courbe de distribution (KDE)
        fig_kde = px.density_contour(
            df, x="MPD", marginal_y="violin", title="MPD – Distribution KDE",
            template="plotly_white"
        )
        fig_kde.update_layout(height=500)

        # Scatter MPD vs n_leaves
        fig_scatter = px.scatter(
            df, x="n_leaves", y="MPD", title="MPD vs n_leaves",
            template="plotly_white",
            hover_data=["MPD", "n_leaves"]
        )
        fig_scatter.update_layout(height=500)

        return html.Div([
            outlier_info,
            dcc.Graph(figure=fig_hist, id="mpd-histogram"),
            dcc.Graph(figure=fig_kde, id="mpd-kde"),
            dcc.Graph(figure=fig_scatter, id="mpd-scatter")
        ])

    # Callback pour afficher les détails du classificateur sélectionné
    @app.callback(
        Output("classifier-details", "children"),
        Input("classifier-dropdown-tab", "value"),
        prevent_initial_call=False
    )
    def display_classifier_details(selected_classifier):
        """Affiche les courbes ROC et learning curves pour le classificateur sélectionné"""
        if not selected_classifier:
            return html.Div("Sélectionnez un classificateur pour voir les détails.")
        
        try:
            # Créer les courbes ROC
            fig_roc = create_roc_curves_plotly(selected_classifier)
            
            # Créer les courbes d'apprentissage
            fig_learning = create_learning_curves_plotly(selected_classifier)
            
            # Retourner un seul div avec un ID unique et une hauteur fixe
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H5(f"Courbes ROC - {selected_classifier}", className="text-center mb-3"),
                        html.Div([
                            dcc.Graph(
                                id=f"roc-graph-{selected_classifier}",
                                figure=fig_roc,
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '500px'}
                            ),
                        ], style={'height': '500px'}),
                    ], width=6),
                    dbc.Col([
                        html.H5(f"Courbes d'apprentissage - {selected_classifier}", className="text-center mb-3"),
                        html.Div([
                            dcc.Graph(
                                id=f"learning-graph-{selected_classifier}",
                                figure=fig_learning,
                                config={'displayModeBar': True, 'displaylogo': False},
                                style={'height': '600px'}
                            ),
                        ], style={'height': '600px'}),
                    ], width=6),
                ], className="mt-3"),
            ], id=f"classif-details-container-{selected_classifier}", style={'minHeight': '650px', 'maxHeight': '650px'})
        except Exception as e:
            return html.Div([
                html.Div(f"Erreur lors du chargement des détails pour {selected_classifier}: {str(e)}", className="alert alert-danger"),
                html.Pre(traceback.format_exc(), style={"fontSize": "10px"})
            ], id=f"error-container-{selected_classifier}")

    # Flask route to serve tree files for download
    @server.route("/download-tree/<path:filename>")
    def download_tree(filename):
        p = results_path() / "trees" / filename
        if not p.exists():
            return flask.abort(404)
        return flask.send_file(str(p), as_attachment=True, download_name=filename)

    return app, server


def run_dashboard():
    """Lance le dashboard Dash avec gestion d'erreurs robuste"""
    try:
        app, server = make_app()
        print(f"\nDashboard démarré sur http://127.0.0.1:8050")
        print("Appuyez sur Ctrl+C pour arrêter le serveur\n")
        
        # Lancer le serveur avec gestion d'erreurs
        app.run(debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nArrêt du dashboard demandé par l'utilisateur (Ctrl+C)")
        print("Dashboard fermé proprement.")
    except Exception as e:
        print(f"\nErreur fatale dans le dashboard: {e}")
        import traceback
        traceback.print_exc()
        print("\nLe dashboard va se fermer.")
        raise


if __name__ == "__main__":
    run_dashboard()
