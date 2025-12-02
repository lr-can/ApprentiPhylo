"""
data_description2.py
============================================
Génère des histogrammes en mémoire (matplotlib figure) au lieu d’écrire des PNG.

Retourne un dictionnaire de la forme :
{
    "file1.tsv": { "identity": fig, "gap": fig, ... },
    "file2.tsv": { ... },
    "[COMBINÉ Réelles]": { ... },
    "[COMBINÉ Simulées]": { ... }
}
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


COLUMNS = ["identity", "gap", "length", "nseq"]


def load_tsv(file_path):
    df = pd.read_csv(file_path, sep="\t")
    return df


def make_histogram(data, column, density=False):
    """Crée une figure matplotlib en mémoire."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data[column].dropna(), bins=20, color="skyblue", edgecolor="black", density=density)
    ax.set_title(f"Histogramme de {column}" + (" (densité normalisée)" if density else ""))
    ax.set_xlabel(column)
    ax.set_ylabel("Densité" if density else "Fréquence")
    fig.tight_layout()
    return fig


def describe_data_figures(files, label_combined=None, density=False):
    """
    Retourne :
    - un dict de figures pour chaque fichier
    - un dict supplémentaire pour la version combinée

    files : liste des chemins
    label_combined : string à afficher (ex : "[COMBINÉ Réelles]")
    density : True = histogramme normalisé
    """
    results = {}

    dfs = []
    for f in files:
        df = load_tsv(f)
        dfs.append(df)

        figs_for_file = {}
        for col in COLUMNS:
            if col in df.columns:
                figs_for_file[col] = make_histogram(df, col, density=density)

        results[str(Path(f).name)] = figs_for_file

    # Ajout de l’ensemble fusionné
    if label_combined is not None:
        df_all = pd.concat(dfs, ignore_index=True)
        combined_figs = {}
        for col in COLUMNS:
            if col in df_all.columns:
                combined_figs[col] = make_histogram(df_all, col, density=density)

        results[label_combined] = combined_figs

    return results
