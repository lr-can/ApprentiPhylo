"""
data_description.py
====================
Produit des statistiques descriptives et des histogrammes sur les alignements (TSV/FASTA).  
Calcule des distributions (longueur, gaps, identité, nombre de séquences)  
et permet de visualiser la structure des données d’entrée ou de sortie.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def setup_visualization():
    """Configure le style de visualisation."""
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
            data = pd.read_csv(file_path, sep='\t')
            if col in data.columns:
                col_min = min(col_min, data[col].min())
                col_max = max(col_max, data[col].max())
        global_min[col] = col_min
        global_max[col] = col_max
    return global_min, global_max


def generate_histograms(file_path, columns, global_min, global_max, bins=20):
    """Génère et enregistre des histogrammes pour chaque colonne."""
    data = pd.read_csv(file_path, sep='\t')
    base_name = Path(file_path).stem
    output_dir = Path(f"{base_name}_plots")
    output_dir.mkdir(exist_ok=True)

    for col in columns:
        if col in data.columns:
            plt.figure()
            plt.hist(
                data[col],
                bins=bins,
                range=(global_min[col], global_max[col]),
                color='blue',
                alpha=0.7,
                edgecolor='black'
            )
            plt.title(f"Histogramme de {col}")
            plt.xlabel("Valeurs")
            plt.ylabel("Fréquence")
            plt.xlim(global_min[col], global_max[col])
            plt.savefig(output_dir / f"histogram_{col}.png")
            plt.close()


def describe_data(files):
    """Pipeline complet pour générer les histogrammes."""
    setup_visualization()
    columns = ["identity", "gap", "length", "nseq"]
    global_min, global_max = determine_global_range(files, columns)
    for file in files:
        generate_histograms(file, columns, global_min, global_max)
