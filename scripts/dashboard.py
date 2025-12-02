import os
from pathlib import Path
import base64
import io
import sys

from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Imports de ton backend ---
from scripts.data_description2 import describe_data_figures
from scripts.phylo_metrics import tree_summary
from scripts.analyse_classif import (
    load_and_concat_parquets,
    plot_learning_curves,
    plot_roc_curves
)

# --- Configuration générale ---
UPLOAD_FOLDER = Path("uploaded_files")
UPLOAD_FOLDER.mkdir(exist_ok=True)
REPORT_FOLDER = Path("reports")
REPORT_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER = Path("results/classification")
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "ApprentiPhylo Dashboard"

# --------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------
app.layout = dbc.Container([
    html.H2("🧬 PhyloClassifier", className="text-center mt-4 mb-4"),

    dbc.Row([
        dbc.Col([
            html.H4("Import des données", className="mb-3"),
            html.P("Séquences réelles (FASTA)"),
            dcc.Upload(
                id="upload-real-data",
                children=html.Div(['⬆️ Glisser ou ', html.B('choisir les fichiers de séquences empiriques')]),
                multiple=True,
                className="border p-3 text-center rounded bg-light mb-2",
            ),
            html.P("Séquences simulées (FASTA)"),
            dcc.Upload(
                id="upload-simul-data",
                children=html.Div(['⬆️ Glisser ou ', html.B('choisir les fichiers de séquences simulées')]),
                multiple=True,
                className="border p-3 text-center rounded bg-light mb-2",
            ),
            html.P("Arbres phylogénétiques (.nwk ou .nw)"),
            dcc.Upload(
                id="upload-arbres",
                children=html.Div(['🌳 Glisser ou ', html.B('choisir un arbre Newick')]),
                multiple=True,
                className="border p-3 text-center rounded bg-light mb-3",
            ),
        ], width=4),

        dbc.Col([
            html.H4("Statistiques de base", className="mb-3"),
            html.Div(id="output-stats-reelles", className="alert alert-secondary"),
            html.Div(id="output-stats-simulees", className="alert alert-secondary"),
            html.Div(id="output-arbres", className="alert alert-secondary"),
        ], width=8)
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.H4("Histogrammes des séquences", className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Fichier :"),
                    dcc.Dropdown(id="dropdown-file-left", placeholder="Choisir un fichier"),
                    html.Label("Variable :"),
                    dcc.Dropdown(id="dropdown-var-left", placeholder="Choisir un histogramme"),
                    dcc.Graph(id="graph-left"),
                    dbc.Checklist(
                        id="switch-density-left",
                        options=[{"label": "Normaliser la densité", "value": "density"}],
                        value=[],
                        switch=True,
                        className="mt-2"
                    ),
                ], width=6),

                dbc.Col([
                    html.Label("Fichier :"),
                    dcc.Dropdown(id="dropdown-file-right", placeholder="Choisir un fichier"),
                    html.Label("Variable :"),
                    dcc.Dropdown(id="dropdown-var-right", placeholder="Choisir un histogramme"),
                    dcc.Graph(id="graph-right"),
                    dbc.Checklist(
                        id="switch-density-right",
                        options=[{"label": "Normaliser la densité", "value": "density"}],
                        value=[],
                        switch=True,
                        className="mt-2"
                    ),
                ], width=6),
            ], className="mt-3"),

            dbc.Button("Télécharger histogramme gauche", id="btn-dl-left", color="secondary", className="mt-3"),
            dbc.Button("Télécharger histogramme droit", id="btn-dl-right", color="secondary", style={"margin-left": "10px"}, className="mt-3"),

            dcc.Download(id="download-histo-left"),
            dcc.Download(id="download-histo-right")
        ], width=12)
    ]),

    html.Hr(),

    # --- SECTION : Résultats de classification ---
    dbc.Row([
        dbc.Col([
            html.H4("📊 Résultats de classification", className="mb-3"),
            html.P("Charger les résultats depuis le dossier results/classification"),
            dbc.Button("🔄 Charger les résultats", id="btn-load-results", color="info", className="mb-3"),
            html.Div(id="loading-status", className="alert alert-secondary mb-3"),
            
            # Onglets pour organiser les résultats
            dbc.Tabs([
                # Onglet 1: Données d'entraînement
                dbc.Tab(
                    label="📈 Données d'entraînement",
                    children=[
                        html.Div([
                            html.H5("Statistiques par classifieur", className="mt-3 mb-2"),
                            html.Div(id="training-stats", className="mb-3"),
                            html.H5("Aperçu des données complètes", className="mt-3 mb-2"),
                            html.Div(id="training-data-table", className="mb-3", style={"overflowX": "scroll"}),
                        ])
                    ]
                ),
                
                # Onglet 2: Prédictions
                dbc.Tab(
                    label="🎯 Prédictions",
                    children=[
                        html.Div([
                            html.H5("Métriques de performance", className="mt-3 mb-2"),
                            html.Div(id="predictions-metrics", className="mb-3"),
                            html.H5("Aperçu des prédictions", className="mt-3 mb-2"),
                            html.Div(id="predictions-table", className="mb-3", style={"overflowX": "scroll"}),
                        ])
                    ]
                ),
                
                # Onglet 3: Courbes d'apprentissage
                dbc.Tab(
                    label="📉 Courbes d'apprentissage",
                    children=[
                        html.Div([
                            html.H5("Sélectionner une courbe", className="mt-3 mb-2"),
                            dcc.Dropdown(
                                id="dropdown-learning-curve",
                                placeholder="Choisir une courbe d'apprentissage"
                            ),
                            html.Img(id="learning-curve-image", style={"width": "100%", "marginTop": "20px"}),
                        ])
                    ]
                ),
                
                # Onglet 4: Courbes ROC
                dbc.Tab(
                    label="📊 Courbes ROC",
                    children=[
                        html.Div([
                            html.H5("Sélectionner une courbe ROC", className="mt-3 mb-2"),
                            dcc.Dropdown(
                                id="dropdown-roc-curve",
                                placeholder="Choisir une courbe ROC"
                            ),
                            html.Img(id="roc-curve-image", style={"width": "100%", "marginTop": "20px"}),
                        ])
                    ]
                ),
            ]),
        ], width=12)
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.H4("📄 Génération de rapport PDF", className="mb-3"),
            dbc.Button("Générer le rapport PDF", id="btn-rapport", color="primary", className="mb-3"),
            html.Div(id="rapport-resultat", className="alert alert-info")
        ])
    ]),
    
    # Stores pour les données
    dcc.Store(id="stored-figs", data={}),
    dcc.Store(id="stored-report-data", data={}),

], fluid=True)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def save_uploaded_files(contents_list, filenames, subfolder):
    """Sauvegarde les fichiers uploadés localement et retourne leurs chemins."""
    folder = UPLOAD_FOLDER / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for content, name in zip(contents_list, filenames):
        data = content.encode("utf8").split(b";base64,")[1]
        file_path = folder / name
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(data))
        saved_paths.append(file_path)
    return saved_paths

def fig_to_base64(fig):
    """Convertit une figure matplotlib en base64."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode()

def image_to_base64(image_path):
    """Convertit une image sur disque en base64."""
    if not Path(image_path).exists():
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded}"


# --------------------------------------------------------------------
# Callbacks pour les résultats de classification
# --------------------------------------------------------------------

@app.callback(
    [Output("stored-report-data", "data"),
     Output("loading-status", "children")],
    Input("btn-load-results", "n_clicks"),
    prevent_initial_call=True
)
def load_classification_results(n_clicks):
    """Charge les résultats de classification depuis le dossier results/classification."""
    if not n_clicks:
        return {}, "Cliquez sur le bouton pour charger les résultats."
    
    try:
        base_dir = RESULTS_FOLDER
        output_dir = base_dir / "plots"
        output_dir.mkdir(exist_ok=True)
        
        classifiers = [
            "AACnnClassifier",
            "DenseMsaClassifier",
            "DenseSiteClassifier",
            "LogisticRegressionClassifier"
        ]
        
        # Charger les données
        train_df = load_and_concat_parquets(base_dir, classifiers, "train_history.parquet")
        pred_df = load_and_concat_parquets(base_dir, classifiers, "best_preds.parquet")
        
        if train_df.empty and pred_df.empty:
            return {}, "❌ Aucune donnée trouvée dans results/classification"
        
        # Générer les courbes d'apprentissage
        learning_curve_paths = plot_learning_curves(train_df, output_dir)
        
        # Générer les courbes ROC
        roc_curve_paths = plot_roc_curves(base_dir, output_dir)
        
        # Préparer les données pour le store
        report_data = {
            "train_df": train_df.to_dict('records') if not train_df.empty else [],
            "pred_df": pred_df.to_dict('records') if not pred_df.empty else [],
            "learning_curves": [str(p) for p in learning_curve_paths],
            "roc_curves": [str(p) for p in roc_curve_paths],
        }
        
        status = f"✅ Données chargées avec succès : {len(train_df)} lignes d'entraînement, {len(pred_df)} prédictions, {len(learning_curve_paths)} courbes d'apprentissage, {len(roc_curve_paths)} courbes ROC"
        
        return report_data, status
        
    except Exception as e:
        return {}, f"❌ Erreur lors du chargement : {str(e)}"


@app.callback(
    Output("training-stats", "children"),
    Input("stored-report-data", "data")
)
def display_training_stats(report_data):
    """Affiche les statistiques des données d'entraînement."""
    if not report_data or not report_data.get("train_df"):
        return "Aucune donnée d'entraînement disponible."
    
    try:
        df = pd.DataFrame(report_data["train_df"])
        
        # Calculer les statistiques par classifieur
        stats_list = []
        
        for clf in df["classifier"].unique():
            clf_df = df[df["classifier"] == clf]
            
            # Grouper par run si la colonne existe
            if "run" in clf_df.columns:
                for run in clf_df["run"].unique():
                    run_df = clf_df[clf_df["run"] == run]
                    
                    # Trouver la meilleure époque
                    if "val_loss" in run_df.columns:
                        best_epoch = run_df.loc[run_df["val_loss"].idxmin()]
                    else:
                        best_epoch = run_df.iloc[-1]
                    
                    stats_list.append({
                        "Classifieur": clf,
                        "Run": run,
                        "Meilleure époque": int(best_epoch.get("epoch", 0)),
                        "Val Loss": f"{best_epoch.get('val_loss', 0):.4f}",
                        "Val Acc": f"{best_epoch.get('val_acc', 0):.4f}",
                        "F1": f"{best_epoch.get('f1', 0):.4f}",
                    })
            else:
                # Sans colonne run
                if "val_loss" in clf_df.columns:
                    best_epoch = clf_df.loc[clf_df["val_loss"].idxmin()]
                else:
                    best_epoch = clf_df.iloc[-1]
                
                stats_list.append({
                    "Classifieur": clf,
                    "Meilleure époque": int(best_epoch.get("epoch", 0)),
                    "Val Loss": f"{best_epoch.get('val_loss', 0):.4f}",
                    "Val Acc": f"{best_epoch.get('val_acc', 0):.4f}",
                    "F1": f"{best_epoch.get('f1', 0):.4f}",
                })
        
        stats_df = pd.DataFrame(stats_list)
        
        return dbc.Table.from_dataframe(
            stats_df,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size="sm"
        )
        
    except Exception as e:
        return f"Erreur lors du calcul des statistiques : {str(e)}"


@app.callback(
    Output("training-data-table", "children"),
    Input("stored-report-data", "data")
)
def display_training_table(report_data):
    """Affiche le tableau complet des données d'entraînement."""
    if not report_data or not report_data.get("train_df"):
        return "Aucune donnée d'entraînement disponible."
    
    try:
        df = pd.DataFrame(report_data["train_df"])
        
        # Limiter à 50 premières lignes pour la performance
        display_df = df.head(50)
        
        return html.Div([
            dbc.Table.from_dataframe(
                display_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size="sm"
            ),
            html.P(f"Affichage des 50 premières lignes sur {len(df)} total", className="text-muted small")
        ])
        
    except Exception as e:
        return f"Erreur : {str(e)}"


@app.callback(
    Output("predictions-metrics", "children"),
    Input("stored-report-data", "data")
)
def display_predictions_metrics(report_data):
    """Affiche les métriques de performance des prédictions."""
    if not report_data or not report_data.get("pred_df"):
        return "Aucune donnée de prédiction disponible."
    
    try:
        df = pd.DataFrame(report_data["pred_df"])
        
        # Calculer les métriques par classifieur
        metrics_list = []
        
        # Déterminer les colonnes disponibles
        pred_col = "pred" if "pred" in df.columns else "pred_label"
        target_col = "target" if "target" in df.columns else "true_label"
        
        if pred_col not in df.columns or target_col not in df.columns:
            return "Colonnes de prédiction non trouvées dans les données."
        
        for clf in df["classifier"].unique():
            clf_df = df[df["classifier"] == clf]
            
            # Grouper par run si la colonne existe
            if "run" in clf_df.columns:
                for run in clf_df["run"].unique():
                    run_df = clf_df[clf_df["run"] == run]
                    
                    accuracy = (run_df[pred_col] == run_df[target_col]).mean()
                    
                    metrics_list.append({
                        "Classifieur": clf,
                        "Run": run,
                        "Précision": f"{accuracy:.4f}",
                        "Nb prédictions": len(run_df),
                    })
            else:
                accuracy = (clf_df[pred_col] == clf_df[target_col]).mean()
                
                metrics_list.append({
                    "Classifieur": clf,
                    "Précision": f"{accuracy:.4f}",
                    "Nb prédictions": len(clf_df),
                })
        
        metrics_df = pd.DataFrame(metrics_list)
        
        return dbc.Table.from_dataframe(
            metrics_df,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            size="sm"
        )
        
    except Exception as e:
        return f"Erreur lors du calcul des métriques : {str(e)}"


@app.callback(
    Output("predictions-table", "children"),
    Input("stored-report-data", "data")
)
def display_predictions_table(report_data):
    """Affiche le tableau des prédictions."""
    if not report_data or not report_data.get("pred_df"):
        return "Aucune donnée de prédiction disponible."
    
    try:
        df = pd.DataFrame(report_data["pred_df"])
        
        # Limiter à 50 premières lignes
        display_df = df.head(50)
        
        return html.Div([
            dbc.Table.from_dataframe(
                display_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size="sm"
            ),
            html.P(f"Affichage des 50 premières lignes sur {len(df)} total", className="text-muted small")
        ])
        
    except Exception as e:
        return f"Erreur : {str(e)}"


@app.callback(
    Output("dropdown-learning-curve", "options"),
    Input("stored-report-data", "data")
)
def fill_learning_curve_dropdown(report_data):
    """Remplit le dropdown des courbes d'apprentissage."""
    if not report_data or not report_data.get("learning_curves"):
        return []
    
    curves = report_data["learning_curves"]
    return [{"label": Path(c).name, "value": c} for c in curves]


@app.callback(
    Output("dropdown-roc-curve", "options"),
    Input("stored-report-data", "data")
)
def fill_roc_curve_dropdown(report_data):
    """Remplit le dropdown des courbes ROC."""
    if not report_data or not report_data.get("roc_curves"):
        return []
    
    curves = report_data["roc_curves"]
    return [{"label": Path(c).name, "value": c} for c in curves]


@app.callback(
    Output("learning-curve-image", "src"),
    Input("dropdown-learning-curve", "value")
)
def display_learning_curve(curve_path):
    """Affiche la courbe d'apprentissage sélectionnée."""
    if not curve_path:
        return ""
    return image_to_base64(curve_path)


@app.callback(
    Output("roc-curve-image", "src"),
    Input("dropdown-roc-curve", "value")
)
def display_roc_curve(curve_path):
    """Affiche la courbe ROC sélectionnée."""
    if not curve_path:
        return ""
    return image_to_base64(curve_path)


# --------------------------------------------------------------------
# Callbacks pour les données de séquences
# --------------------------------------------------------------------

@app.callback(
    Output("output-stats-reelles", "children"),
    Input("upload-real-data", "contents"),
    State("upload-real-data", "filename")
)
def afficher_stats_reelles(contents, filenames):
    if not contents:
        return "Aucun fichier réel uploadé."
    saved_paths = save_uploaded_files(contents, filenames, "reelles")
    return html.Ul([html.Li(f"✅ {len(saved_paths)} fichier(s) réel(s) uploadé(s)")])


@app.callback(
    Output("output-stats-simulees", "children"),
    Input("upload-simul-data", "contents"),
    State("upload-simul-data", "filename")
)
def afficher_stats_simulees(contents, filenames):
    if not contents:
        return "Aucun fichier simulé uploadé."
    saved_paths = save_uploaded_files(contents, filenames, "simulees")
    return html.Ul([html.Li(f"✅ {len(saved_paths)} fichier(s) simulé(s) uploadé(s)")])


@app.callback(
    Output("stored-figs", "data"),
    Input("upload-real-data", "contents"),
    State("upload-real-data", "filename"),
    Input("upload-simul-data", "contents"),
    State("upload-simul-data", "filename"),
)
def load_and_store_figs(real_contents, real_names, simul_contents, simul_names):
    if not real_contents and not simul_contents:
        return {}

    all_results = {}

    if real_contents:
        real_paths = save_uploaded_files(real_contents, real_names, "reelles")
        figs_real = describe_data_figures(
            files=[str(p) for p in real_paths],
            label_combined="[COMBINÉ Réelles]",
            density=False
        )
        all_results["real"] = figs_real

    if simul_contents:
        simul_paths = save_uploaded_files(simul_contents, simul_names, "simulees")
        figs_sim = describe_data_figures(
            files=[str(p) for p in simul_paths],
            label_combined="[COMBINÉ Simulées]",
            density=False
        )
        all_results["simul"] = figs_sim

    return all_results


@app.callback(
    Output("dropdown-file-left", "options"),
    Input("stored-figs", "data")
)
def fill_left_files(data):
    if not data:
        return []
    options = []
    for bloc in data.values():
        for name in bloc.keys():
            options.append({"label": name, "value": name})
    return options


@app.callback(
    Output("dropdown-file-right", "options"),
    Input("stored-figs", "data")
)
def fill_right_files(data):
    if not data:
        return []
    options = []
    for bloc in data.values():
        for name in bloc.keys():
            options.append({"label": name, "value": name})
    return options


VAR_LIST = ["identity", "gap", "length", "nseq"]

@app.callback(
    Output("dropdown-var-left", "options"),
    Input("dropdown-file-left", "value")
)
def fill_var_left(file_selected):
    if not file_selected:
        return []
    return [{"label": v, "value": v} for v in VAR_LIST]


@app.callback(
    Output("dropdown-var-right", "options"),
    Input("dropdown-file-right", "value")
)
def fill_var_right(file_selected):
    if not file_selected:
        return []
    return [{"label": v, "value": v} for v in VAR_LIST]


@app.callback(
    Output("graph-left", "figure"),
    Input("dropdown-file-left", "value"),
    Input("dropdown-var-left", "value"),
    Input("switch-density-left", "value"),
    State("stored-figs", "data")
)
def update_graph_left(file_name, var, density_switch, data):
    from plotly.tools import mpl_to_plotly

    if not data or not file_name or not var:
        return {}

    for bloc in data.values():
        if file_name in bloc:
            fig = bloc[file_name][var]
            break
    else:
        return {}

    return mpl_to_plotly(fig)


@app.callback(
    Output("graph-right", "figure"),
    Input("dropdown-file-right", "value"),
    Input("dropdown-var-right", "value"),
    Input("switch-density-right", "value"),
    State("stored-figs", "data")
)
def update_graph_right(file_name, var, density_switch, data):
    from plotly.tools import mpl_to_plotly

    if not data or not file_name or not var:
        return {}

    for bloc in data.values():
        if file_name in bloc:
            fig = bloc[file_name][var]
            break
    else:
        return {}

    return mpl_to_plotly(fig)


@app.callback(
    Output("output-arbres", "children"),
    Input("upload-arbres", "contents"),
    State("upload-arbres", "filename")
)
def afficher_mpd(contents, filenames):
    if not contents:
        return "Aucun arbre uploadé."
    saved_paths = save_uploaded_files(contents, filenames, "arbres")
    try:
        results = [tree_summary(str(p)) for p in saved_paths]
        items = [html.Li(f"{f.name} : MPD={r['MPD']:.4f}, feuilles={r['n_leaves']}") 
                for f, r in zip(saved_paths, results)]
        return html.Ul(items)
    except Exception as e:
        return f"Erreur lors de l'analyse des arbres : {e}"


@app.callback(
    Output("rapport-resultat", "children"),
    Input("btn-rapport", "n_clicks")
)
def generer_rapport(n_clicks):
    if not n_clicks:
        return "Cliquez sur le bouton pour générer un rapport PDF complet."
    try:
        from scripts.report import process_classification_results
        pdf_path = RESULTS_FOLDER / "report.pdf"
        process_classification_results(
            base_dir=str(RESULTS_FOLDER),
            output_pdf=str(pdf_path)
        )
        return f"✅ Rapport PDF généré : {pdf_path}"
    except Exception as e:
        return f"❌ Erreur lors de la génération du rapport : {e}"


# --------------------------------------------------------------------
# Lancement du serveur
# --------------------------------------------------------------------
def run_dashboard(debug=True, host="127.0.0.1", port=8050):
    """Lance le dashboard Dash."""
    print(f"🚀 Lancement du dashboard sur http://{host}:{port}")
    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()