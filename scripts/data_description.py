"""
data_description.py
====================
Produit des statistiques descriptives et des histogrammes sur les alignements (TSV/FASTA).  
Calcule des distributions (longueur, gaps, identité, nombre de séquences)  
et permet de visualiser la structure des données d’entrée ou de sortie.

Version modifiée :
 - Ne crée plus de fichiers PNG.
 - Retourne des Figures matplotlib prêtes à être affichées dans Dash.
 - Permet à Dash d’enregistrer les plots à la demande.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def setup_visualization():
    """Configure le style des visualisations."""
    try:
        import seaborn as sns
        sns.set(style="whitegrid")
    except ImportError:
        plt.style.use("ggplot")
    plt.rcParams['figure.figsize'] = (10, 6)


def determine_global_range(file_paths, columns):
    """Détermine les bornes globales (min et max) pour chaque colonne."""
    global_min, global_max = {}, {}
    for col in columns:
        col_min, col_max = float('inf'), float('-inf')
        for file_path in file_paths:
            data = pd.read_csv(file_path, sep="\t")
            if col in data.columns:
                col_min = min(col_min, data[col].min())
                col_max = max(col_max, data[col].max())
        global_min[col] = col_min
        global_max[col] = col_max
    return global_min, global_max


def generate_histograms_figures(file_path, columns, global_min, global_max, bins=20):
    """
    Génère des figures matplotlib en mémoire.
    Retourne un dict {colonne: Figure}.
    """
    data = pd.read_csv(file_path, sep="\t")
    figs = {}

    for col in columns:
        if col not in data.columns:
            continue

        fig, ax = plt.subplots()
        ax.hist(
            data[col],
            bins=bins,
            range=(global_min[col], global_max[col]),
            color="blue",
            alpha=0.7,
            edgecolor="black"
        )
        ax.set_title(f"Histogramme de {col}")
        ax.set_xlabel("Valeurs")
        ax.set_ylabel("Fréquence")
        ax.set_xlim(global_min[col], global_max[col])

        figs[col] = fig

    return figs


def describe_data_figures(files):
    """
    Pipeline complet :
      - Calcule les bornes globales
      - Génère toutes les figures
      - Retourne :
          {
             "fichier1.tsv": {"gap": fig1, "length": fig2, ...},
             "fichier2.tsv": {...}
          }
    """
    setup_visualization()
    columns = ["identity", "gap", "length", "nseq"]

    global_min, global_max = determine_global_range(files, columns)

    output = {}
    for file in files:
        figs = generate_histograms_figures(file, columns, global_min, global_max)
        output[file] = figs

    return output
