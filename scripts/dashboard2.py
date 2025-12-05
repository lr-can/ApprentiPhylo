# scripts/dashboard2.py
# Dash dashboard complet avec callbacks et auto-refresh.
import io
import base64
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

REFRESH_INTERVAL_MS = 60000  # auto-refresh every 1 minute


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
    preds_run2_file = base_dir / "run_2" / "preds_run2.parquet"
    preds_run1_file = base_dir / "run_1" / "preds_run1.parquet"
    
    if not preds_run2_file.exists():
        return None
    
    try:
        # Charger les prédictions RUN 2
        if HAS_POLARS:
            df_run2 = pl.read_parquet(preds_run2_file)
            
            if "prob_real" not in df_run2.columns or "filename" not in df_run2.columns:
                return None
            
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
            run2_sim_dir = base_dir / "run_2_sim"
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
            
            if "prob_real" not in df_run2.columns or "filename" not in df_run2.columns:
                return None
            
            # Charger RUN 1
            df_run1 = None
            if preds_run1_file.exists():
                df_run1 = pd.read_parquet(preds_run1_file)
                df_run1 = df_run1.rename(columns={"prob_real": "prob_real_run1", "pred_class": "pred_class_run1"})
            
            # Filtrer simulations
            run2_sim_dir = base_dir / "run_2_sim"
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
        
        for filename in df_pandas["filename"]:
            file_path = run2_sim_dir / filename
            length, n_seqs = get_alignment_length(file_path)
            alignment_lengths.append(length if length is not None else "N/A")
            num_sequences.append(n_seqs if n_seqs is not None else "N/A")
        
        df_pandas["alignment_length"] = alignment_lengths
        df_pandas["num_sequences"] = num_sequences
        
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
        cols_order.extend(["alignment_length", "num_sequences"])
        
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
    
    # Chercher le threshold utilisé dans les logs
    log_file = list(base_dir.glob("pipeline_*.log"))
    if log_file:
        try:
            with open(log_file[0], 'r') as f:
                content = f.read()
                # Chercher le pattern pour le threshold
                match = re.search(r"optimal threshold = ([\d.]+)", content)
                if match:
                    stats["run1_threshold"] = match.group(1)
        except:
            pass
    
    # Données après run 2 : extraire depuis les logs
    # Le nombre de fichiers gardés est loggé comme "[RUN 2] Selected X sims flagged REAL"
    if log_file:
        try:
            with open(log_file[0], 'r') as f:
                content = f.read()
                # Chercher le pattern pour RUN 2
                match = re.search(r"\[RUN 2\] Selected (\d+) sims flagged REAL", content)
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
    
    return stats


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
                    line=dict(color=colors[f"run_{run_num}"], width=2)
                ))
        
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

                    # Histogramme MPD
                    fig_hist = px.histogram(
                        df, x="MPD", nbins=40, title="MPD – Histogramme",
                        marginal="box"
                    )

                    # Courbe de distribution (KDE)
                    fig_kde = px.density_contour(
                        df, x="MPD", marginal_y="violin", title="MPD – Distribution KDE"
                    )


                    # Scatter MPD vs n_leaves
                    fig_scatter = px.scatter(
                        df, x="n_leaves", y="MPD", title="MPD vs n_leaves"
                    )

                    return html.Div([
                        dcc.Graph(figure=fig_hist),
                        dcc.Graph(figure=fig_kde),
                        dcc.Graph(figure=fig_scatter)
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
                                    html.P([html.Strong("Simulées gardées: "), f"{run_stats['run2_kept_sim']}"]),
                                    html.P([html.Strong("% depuis RUN 1: "), f"{run_stats['run2_kept_percentage_from_run1']:.2f}%"]),
                                    html.P([html.Strong("% depuis initial: "), f"{run_stats['run2_kept_percentage_from_initial']:.2f}%"]),
                                    html.P([html.Strong("Threshold: "), f"{run_stats.get('run2_threshold', 'N/A')}"]),
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
                # SECTION 1: VIOLIN PLOTS (du meilleur classificateur)
                # ==========================================
                violin_file = predictions_dir / "violin_plots_with_boxes.html"
                if violin_file.exists():
                    # Lire le contenu HTML et l'intégrer directement
                    with open(violin_file, 'r', encoding='utf-8') as f:
                        violin_html = f.read()
                    
                    children.extend([
                        html.H4(f"Distribution des probabilités (Meilleur: {best_clf})", className="mb-3"),
                        html.P("Violin plots avec box plots intégrés montrant la distribution des probabilités prédites pour les alignements réels vs simulés.", className="text-muted"),
                        html.Div(
                            html.Iframe(
                                srcDoc=violin_html,
                                style={"width": "100%", "height": "650px", "border": "1px solid #ddd"}
                            )
                        ),
                        html.Hr(className="my-4"),
                    ])
                else:
                        children.append(html.Div(f"Violin plots non trouvés: {violin_file}", className="alert alert-warning"))
                
                # ==========================================
                # SECTION 2: HISTOGRAMMES ET BOXPLOTS (du meilleur)
                # ==========================================
                children.append(html.H4(f"Analyses complémentaires - {best_clf}", className="mb-3"))
                
                histogram_file = predictions_dir / "histograms_interactive.html"
                boxplot_file = predictions_dir / "boxplots_comparison_interactive.html"
                
                hist_content = None
                box_content = None
                
                if histogram_file.exists():
                    with open(histogram_file, 'r', encoding='utf-8') as f:
                        hist_content = f.read()
                
                if boxplot_file.exists():
                    with open(boxplot_file, 'r', encoding='utf-8') as f:
                        box_content = f.read()
                
                if hist_content and box_content:
                    children.extend([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Histogrammes", className="text-center mb-2"),
                                html.Iframe(
                                    srcDoc=hist_content,
                                    style={"width": "100%", "height": "500px", "border": "1px solid #ddd"}
                                ),
                            ], width=6),
                            dbc.Col([
                                html.H5("Comparaison Boxplots", className="text-center mb-2"),
                                html.Iframe(
                                    srcDoc=box_content,
                                    style={"width": "100%", "height": "500px", "border": "1px solid #ddd"}
                                ),
                            ], width=6),
                        ], className="mb-4"),
                        html.Hr(className="my-4"),
                    ])
                else:
                    if not histogram_file.exists():
                        children.append(html.Div("Histogrammes non trouvés", className="alert alert-warning mb-2"))
                    if not boxplot_file.exists():
                        children.append(html.Div("Boxplots non trouvés", className="alert alert-warning mb-2"))
                
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
                # SECTION 4: MEILLEURES PRÉDICTIONS RUN 2
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
                
                # Visualisation de l'arbre avec Bio.Phylo
                if HAS_PHYLO:
                    try:
                        tree = Phylo.read(str(p), "newick")
                        # Compter les feuilles
                        n_leaves = len(tree.get_terminals())
                        
                        # Ajuster la taille de la figure en fonction du nombre de feuilles
                        # Plus il y a de feuilles, plus la figure doit être haute
                        base_height = 8
                        height_per_leaf = 0.3  # 0.3 inch par feuille
                        fig_height = max(base_height, n_leaves * height_per_leaf)
                        fig_width = min(14, 10 + (n_leaves / 20))  # Largeur adaptative aussi
                        
                        # Dessiner l'arbre
                        figfile = io.BytesIO()
                        plt.figure(figsize=(fig_width, fig_height))
                        Phylo.draw(tree, do_show=False)
                        plt.tight_layout()
                        plt.savefig(figfile, format="png", bbox_inches="tight", dpi=150)
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
                                    ], color="info", className="mb-3"),
                                    html.Img(
                                        src="data:image/png;base64," + encoded,
                                        style={"maxWidth": "100%", "border": "1px solid #dee2e6", "borderRadius": "5px"}
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
                            "Installez biopython et matplotlib pour activer la visualisation des arbres."
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
    app, server = make_app()
    app.run(debug=True)


if __name__ == "__main__":
    run_dashboard()
