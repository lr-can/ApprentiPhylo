import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configurer les styles de visualisation
def setup_visualization():
    try:
        import seaborn as sns  # Vérifie si Seaborn est installé
        sns.set(style="whitegrid")
    except ImportError:
        # Utiliser un style Matplotlib par défaut si Seaborn n'est pas disponible
        plt.style.use("ggplot")
    plt.rcParams['figure.figsize'] = (10, 6)

def determine_global_range(file_paths, columns):
    """Détermine les limites globales des colonnes spécifiées pour tous les fichiers."""
    global_min = {}
    global_max = {}

    for col in columns:
        col_min = float('inf')
        col_max = float('-inf')
        for file_path in file_paths:
            data = pd.read_csv(file_path, sep='\t')
            if col in data.columns:
                col_min = min(col_min, data[col].min())
                col_max = max(col_max, data[col].max())
        global_min[col] = col_min
        global_max[col] = col_max

    return global_min, global_max

def generate_histograms(file_path, columns, global_min, global_max, bins=20):
    """Génère et enregistre des histogrammes pour les colonnes spécifiées."""
    # Charger les données
    data = pd.read_csv(file_path, sep='\t')

    # Créer un dossier pour enregistrer les figures
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"{base_name}_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Générer un histogramme pour chaque colonne
    for col in columns:
        if col in data.columns:
            plt.figure()
            plt.hist(data[col], bins=bins, range=(global_min[col], global_max[col]), color='blue', alpha=0.7, edgecolor='black')
            plt.title(f"Histogramme de {col}")
            plt.xlabel("Valeurs")
            plt.ylabel("Fréquence")
            plt.xlim(global_min[col], global_max[col])
            plt.savefig(os.path.join(output_dir, f"histogram_{col}.png"))
            plt.close()

# Configurer l'environnement
setup_visualization()

# Traiter chaque fichier passé en argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_histograms.py data1.tsv data2.tsv ...")
        sys.exit(1)

    # Colonnes à analyser
    columns_of_interest = ["identity", "gap", "length", "nseq"]

    # Étape 1 : Déterminer les limites globales par colonne
    global_min, global_max = determine_global_range(sys.argv[1:], columns_of_interest)

    # Étape 2 : Générer les histogrammes pour chaque fichier
    for file_path in sys.argv[1:]:
        generate_histograms(file_path, columns_of_interest, global_min, global_max, bins=20)
