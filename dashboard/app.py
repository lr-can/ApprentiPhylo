import os
from pathlib import Path
import base64
import io
import sys

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- Imports de ton backend ---
from scripts.data_description import describe_data
from scripts.phylo_metrics import tree_summary
from scripts.report import generate_pdf_report

# --- Configuration générale ---
UPLOAD_FOLDER = Path("uploaded_files")
UPLOAD_FOLDER.mkdir(exist_ok=True)
REPORT_FOLDER = Path("reports")
REPORT_FOLDER.mkdir(exist_ok=True)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "ApprentiPhylo Dashboard"


app.layout = dbc.Container([
    html.H2("🧬 PhyloClassifier", className="text-center mt-4 mb-4"),

    dbc.Row([
        dbc.Col([
            html.H4("1️⃣ Import des données", className="mb-3"),
            html.P("Séquences réelles (FASTA)"),
            dcc.Upload(
                id="upload-reelles",
                children=html.Div(['⬆️ Glisser ou ', html.B('choisir les fichiers de séquences empiriques')]),
                multiple=True,
                className="border p-3 text-center rounded bg-light mb-2",
            ),
            html.P("Séquences simulées (FASTA)"),
            dcc.Upload(
                id="upload-simulees",
                children=html.Div(['⬆️ Glisser ou ', html.B('choisir les fichiers de séquence simulées')]),
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
            html.H4("📊 Statistiques de base", className="mb-3"),
            html.Div(id="output-stats-reelles", className="alert alert-secondary"),
            html.Div(id="output-stats-simulees", className="alert alert-secondary"),
            html.Div(id="output-arbres", className="alert alert-secondary"),
        ], width=8)
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.H4("2️⃣ Évaluation et rapport", className="mb-3"),
            dbc.Button("Générer le rapport PDF", id="btn-rapport", color="primary", className="mb-3"),
            html.Div(id="rapport-resultat", className="alert alert-info")
        ])
    ])
], fluid=True)

# --------------------------------------------------------------------
# 🧠 Fonctions utilitaires
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


# --------------------------------------------------------------------
# 📈 Callbacks : analyse des séquences
# --------------------------------------------------------------------
@app.callback(
    Output("output-stats-reelles", "children"),
    Input("upload-reelles", "contents"),
    State("upload-reelles", "filename")
)
def afficher_stats_reelles(contents, filenames):
    if not contents:
        return "Aucun fichier réel uploadé."
    saved_paths = save_uploaded_files(contents, filenames, "reelles")
    try:
        describe_data([str(p) for p in saved_paths])
        return html.Ul([html.Li(f"Analyse effectuée sur {len(saved_paths)} fichier(s) réel(s).")])
    except Exception as e:
        return f"Erreur lors de l'analyse des séquences réelles : {e}"


@app.callback(
    Output("output-stats-simulees", "children"),
    Input("upload-simulees", "contents"),
    State("upload-simulees", "filename")
)
def afficher_stats_simulees(contents, filenames):
    if not contents:
        return "Aucun fichier simulé uploadé."
    saved_paths = save_uploaded_files(contents, filenames, "simulees")
    try:
        describe_data([str(p) for p in saved_paths])
        return html.Ul([html.Li(f"Analyse effectuée sur {len(saved_paths)} fichier(s) simulé(s).")])
    except Exception as e:
        return f"Erreur lors de l'analyse des séquences simulées : {e}"


# --------------------------------------------------------------------
# 🌳 Callbacks : arbres phylogénétiques
# --------------------------------------------------------------------
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
        items = [html.Li(f"{f.name} : MPD={r['MPD']:.4f}, feuilles={r['n_leaves']}") for f, r in zip(saved_paths, results)]
        return html.Ul(items)
    except Exception as e:
        return f"Erreur lors de l'analyse des arbres : {e}"


# --------------------------------------------------------------------
# 🧾 Callback : génération du rapport PDF
# --------------------------------------------------------------------
@app.callback(
    Output("rapport-resultat", "children"),
    Input("btn-rapport", "n_clicks")
)
def generer_rapport(n_clicks):
    if not n_clicks:
        return "Cliquez sur le bouton pour générer un rapport."
    try:
        pdf_path = generate_pdf_report(simulation_folder=UPLOAD_FOLDER, output_dir=REPORT_FOLDER)
        return f"Rapport généré : {pdf_path}"
    except Exception as e:
        return f"Erreur lors de la génération du rapport : {e}"


# --------------------------------------------------------------------
# 🚀 Lancement du serveur
# --------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

