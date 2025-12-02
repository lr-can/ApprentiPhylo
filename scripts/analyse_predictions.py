#!/usr/bin/env python3
"""
Analyse des prédictions du pipeline de classification
Génère des CSV et des violin plots pour visualiser les distributions
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results/classification")
OUTPUT_DIR = RESULTS_DIR / "predictions_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def add_true_labels(df, run_number):
    """Ajoute les vrais labels en fonction des dossiers sources
    
    Stratégie : Les fichiers dans run_2_real sont les vrais réels.
    Tous les autres fichiers dans les prédictions sont simulés.
    
    ⚠️ IMPORTANT: Les prédictions existantes ont été faites avec l'ancienne convention
    où prob_real représentait en fait P(SIMULATED). On inverse pour corriger.
    """
    if df is None:
        return None
    
    # ⚠️ Inverser les prédictions existantes (ancienne convention)
    print(f"⚠️  Correction des prédictions RUN {run_number} (ancienne convention)")
    df = df.with_columns([
        (1.0 - pl.col("prob_real")).alias("prob_real")
    ])
    
    # Utiliser run_2_real comme référence pour tous les vrais réels
    real_dir = RESULTS_DIR / "run_2_real"
    
    # Lister les fichiers réels
    real_files = set()
    if real_dir.exists():
        real_files = {f.name for f in real_dir.glob("*.fasta")}
    
    # Ajouter la colonne label
    # Si le fichier est dans real_files -> label=1 (Real)
    # Sinon -> label=0 (Simulated)
    def get_label(filename):
        return 1 if filename in real_files else 0
    
    df = df.with_columns([
        pl.col("filename").map_elements(get_label, return_dtype=pl.Int64).alias("label")
    ])
    
    return df


def load_predictions():
    """Charge les prédictions des deux runs et ajoute les vrais labels"""
    run1_path = RESULTS_DIR / "run_1/preds_run1.parquet"
    run2_path = RESULTS_DIR / "run_2/preds_run2.parquet"
    
    run1 = pl.read_parquet(run1_path) if run1_path.exists() else None
    run2 = pl.read_parquet(run2_path) if run2_path.exists() else None
    
    # Ajouter les labels vrais
    run1 = add_true_labels(run1, 1)
    run2 = add_true_labels(run2, 2)
    
    return run1, run2


def analyze_retention(run1_df, run2_df):
    """Analyse les taux de rétention"""
    print("\n" + "="*60)
    print("ANALYSE DES TAUX DE RÉTENTION")
    print("="*60 + "\n")
    
    # RUN 1
    if run1_df is not None:
        run1_sim = run1_df.filter(pl.col("label") == 0)  # Simulated
        run1_real = run1_df.filter(pl.col("label") == 1)  # Real
        
        total_run1 = len(run1_df)
        n_sim_run1 = len(run1_sim)
        n_real_run1 = len(run1_real)
        
        print(f"📊 RUN 1 (Prédictions initiales)")
        print(f"   Total alignements : {total_run1}")
        print(f"   • Réels           : {n_real_run1}")
        print(f"   • Simulés         : {n_sim_run1}")
        
        # Compter combien sont prédits comme REAL (proba > threshold)
        if "prob_real" in run1_df.columns:
            sim_flagged_real = run1_sim.filter(pl.col("prob_real") >= 0.5)
            n_sim_flagged = len(sim_flagged_real)
            pct_sim_flagged = (n_sim_flagged / n_sim_run1 * 100) if n_sim_run1 > 0 else 0
            
            print(f"\n   Simulés flaggés REAL : {n_sim_flagged}/{n_sim_run1} ({pct_sim_flagged:.2f}%)")
    
    # RUN 2
    if run2_df is not None:
        run2_sim = run2_df.filter(pl.col("label") == 0)
        run2_real = run2_df.filter(pl.col("label") == 1)
        
        total_run2 = len(run2_df)
        n_sim_run2 = len(run2_sim)
        n_real_run2 = len(run2_real)
        
        print(f"\n📊 RUN 2 (Après filtrage)")
        print(f"   Total alignements : {total_run2}")
        print(f"   • Réels           : {n_real_run2}")
        print(f"   • Simulés         : {n_sim_run2}")
        
        if "prob_real" in run2_df.columns:
            sim_flagged_real_r2 = run2_sim.filter(pl.col("prob_real") >= 0.5)
            n_sim_flagged_r2 = len(sim_flagged_real_r2)
            pct_sim_flagged_r2 = (n_sim_flagged_r2 / n_sim_run2 * 100) if n_sim_run2 > 0 else 0
            
            print(f"\n   Simulés flaggés REAL : {n_sim_flagged_r2}/{n_sim_run2} ({pct_sim_flagged_r2:.2f}%)")
    
    # Comparaison
    if run1_df is not None and run2_df is not None:
        print(f"\n📉 ÉVOLUTION")
        print(f"   Dataset simulés : {n_sim_run1} → {n_sim_run2} (-{n_sim_run1 - n_sim_run2})")
        if "prob_real" in run1_df.columns and "prob_real" in run2_df.columns:
            print(f"   Rétention finale : {n_sim_flagged_r2} alignements simulés (sur {n_sim_run1} initiaux)")
            pct_final = (n_sim_flagged_r2 / n_sim_run1 * 100) if n_sim_run1 > 0 else 0
            print(f"   Taux de rétention global : {pct_final:.2f}%")
    
    print("\n" + "="*60 + "\n")


def export_predictions_csv(run1_df, run2_df):
    """Export des prédictions en CSV"""
    if run1_df is not None:
        csv_path = OUTPUT_DIR / "predictions_run1.csv"
        run1_df.write_csv(csv_path)
        print(f"✓ Prédictions RUN 1 exportées : {csv_path}")
    
    if run2_df is not None:
        csv_path = OUTPUT_DIR / "predictions_run2.csv"
        run2_df.write_csv(csv_path)
        print(f"✓ Prédictions RUN 2 exportées : {csv_path}")


def plot_violin_distributions(run1_df, run2_df):
    """Génère des violin plots pour visualiser les distributions"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot par run
    for idx, (run_df, run_name) in enumerate([(run1_df, "RUN 1"), (run2_df, "RUN 2")]):
        ax = axes[idx]
        
        if run_df is None or "prob_real" not in run_df.columns:
            ax.text(0.5, 0.5, f'Pas de données pour {run_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        sim = run_df.filter(pl.col("label") == 0)["prob_real"].to_numpy()
        real = run_df.filter(pl.col("label") == 1)["prob_real"].to_numpy()
        
        # Violin plot manuel avec matplotlib
        positions = [1, 2]
        data_to_plot = [sim, real]
        colors = ['#ff7f0e', '#2ca02c']
        labels = [f'Simulé (n={len(sim)})', f'Réel (n={len(real)})']
        
        parts = ax.violinplot(data_to_plot, positions=positions, widths=0.7,
                              showmeans=True, showmedians=True, showextrema=True)
        
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Ligne de seuil
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Seuil = 0.5')
        ax.set_title(f"{run_name} - Distribution des probabilités", fontsize=14, fontweight='bold')
        ax.set_ylabel("Probabilité prédite (classe RÉEL)", fontsize=12)
        ax.set_xlabel("Type d'alignement", fontsize=12)
        ax.set_xticks(positions)
        ax.set_xticklabels(['Simulé', 'Réel'])
        ax.legend([labels[0], labels[1], 'Seuil = 0.5'])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "violin_plot_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Violin plot sauvegardé : {output_path}")
    plt.close()
    
    # Plot comparatif box plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    data_to_plot = []
    labels_plot = []
    colors_plot = []
    
    for run_name, run_df in [("RUN 1", run1_df), ("RUN 2", run2_df)]:
        if run_df is None or "prob_real" not in run_df.columns:
            continue
        
        sim = run_df.filter(pl.col("label") == 0)["prob_real"].to_numpy()
        real = run_df.filter(pl.col("label") == 1)["prob_real"].to_numpy()
        
        data_to_plot.extend([sim, real])
        labels_plot.extend([f'{run_name}\nSimulé', f'{run_name}\nRéel'])
        colors_plot.extend(['#ff7f0e', '#2ca02c'])
    
    if data_to_plot:
        positions = list(range(1, len(data_to_plot) + 1))
        parts = ax.violinplot(data_to_plot, positions=positions, widths=0.7,
                              showmeans=True, showmedians=True, showextrema=True)
        
        for pc, color in zip(parts['bodies'], colors_plot):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Seuil = 0.5')
        ax.set_title("Comparaison des distributions de probabilités (RUN 1 vs RUN 2)", 
                     fontsize=16, fontweight='bold')
        ax.set_ylabel("Probabilité prédite (classe RÉEL)", fontsize=13)
        ax.set_xlabel("", fontsize=13)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_plot)
        ax.legend(['Seuil = 0.5'], fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "violin_plot_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Violin plot comparatif sauvegardé : {output_path}")
    plt.close()


def plot_histogram_distributions(run1_df, run2_df):
    """Génère des histogrammes pour visualiser les distributions"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for idx, (run_df, run_name) in enumerate([(run1_df, "RUN 1"), (run2_df, "RUN 2")]):
        if run_df is None or "prob_real" not in run_df.columns:
            continue
        
        ax = axes[idx]
        
        sim = run_df.filter(pl.col("label") == 0)["prob_real"].to_numpy()
        real = run_df.filter(pl.col("label") == 1)["prob_real"].to_numpy()
        
        bins = np.linspace(0, 1, 50)
        ax.hist(sim, bins=bins, alpha=0.6, label=f'Simulé (n={len(sim)})', 
                color='orange', edgecolor='black')
        ax.hist(real, bins=bins, alpha=0.6, label=f'Réel (n={len(real)})', 
                color='green', edgecolor='black')
        
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Seuil = 0.5')
        ax.set_title(f"{run_name} - Distribution des probabilités prédites", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Probabilité prédite (classe RÉEL)", fontsize=12)
        ax.set_ylabel("Fréquence", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "histogram_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Histogrammes sauvegardés : {output_path}")
    plt.close()


def generate_summary_table(run1_df, run2_df):
    """Génère un tableau récapitulatif"""
    summary = []
    
    for run_name, run_df in [("RUN 1", run1_df), ("RUN 2", run2_df)]:
        if run_df is None or "prob_real" not in run_df.columns:
            continue
        
        for label, label_name in [(0, "Simulé"), (1, "Réel")]:
            subset = run_df.filter(pl.col("label") == label)
            preds = subset["prob_real"].to_numpy()
            
            if len(preds) > 0:
                flagged_real = np.sum(preds >= 0.5)
                pct_flagged = (flagged_real / len(preds)) * 100
                
                summary.append({
                    "Run": run_name,
                    "Type": label_name,
                    "N_total": len(preds),
                    "N_flagged_REAL": flagged_real,
                    "Pct_flagged_REAL": f"{pct_flagged:.2f}%",
                    "Prob_mean": f"{np.mean(preds):.4f}",
                    "Prob_median": f"{np.median(preds):.4f}",
                    "Prob_std": f"{np.std(preds):.4f}",
                    "Prob_min": f"{np.min(preds):.4f}",
                    "Prob_max": f"{np.max(preds):.4f}"
                })
    
    if summary:
        summary_df = pl.DataFrame(summary)
        csv_path = OUTPUT_DIR / "summary_statistics.csv"
        summary_df.write_csv(csv_path)
        print(f"✓ Tableau récapitulatif exporté : {csv_path}")
        print("\n" + summary_df.to_pandas().to_string(index=False))


def main():
    print("\n" + "="*60)
    print("ANALYSE DES PRÉDICTIONS DU PIPELINE")
    print("="*60 + "\n")
    
    # Charger les données
    print("📂 Chargement des prédictions...")
    run1_df, run2_df = load_predictions()
    
    if run1_df is None and run2_df is None:
        print("❌ Aucun fichier de prédictions trouvé !")
        return
    
    print(f"   • RUN 1: {len(run1_df) if run1_df is not None else 0} alignements")
    print(f"   • RUN 2: {len(run2_df) if run2_df is not None else 0} alignements")
    
    # Analyses
    analyze_retention(run1_df, run2_df)
    
    print("\n📊 Génération des visualisations...")
    plot_violin_distributions(run1_df, run2_df)
    plot_histogram_distributions(run1_df, run2_df)
    
    print("\n📁 Export des données...")
    export_predictions_csv(run1_df, run2_df)
    generate_summary_table(run1_df, run2_df)
    
    print("\n" + "="*60)
    print("✅ ANALYSE TERMINÉE")
    print(f"📂 Résultats disponibles dans : {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

