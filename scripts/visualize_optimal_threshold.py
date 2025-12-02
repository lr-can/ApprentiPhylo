#!/usr/bin/env python3
"""
Visualise l'impact du calcul du seuil optimal (Youden's J) vs ancien seuil (AUC)
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

# Configuration
RESULTS_DIR = Path("results/classification")
OUTPUT_DIR = RESULTS_DIR / "predictions_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


def load_roc_data(run_number):
    """Charge les données ROC exportées"""
    roc_file = RESULTS_DIR / f"run_{run_number}/roc_data/AACnnClassifier_roc.csv"
    if roc_file.exists():
        return pl.read_csv(roc_file)
    return None


def calculate_optimal_threshold_youden(fpr, tpr, thresholds):
    """Calcule le seuil optimal avec le J de Youden (TPR - FPR)"""
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx], j_scores[optimal_idx], optimal_idx


def load_predictions_with_labels(run_number):
    """Charge les prédictions et ajoute les vrais labels"""
    preds_file = RESULTS_DIR / f"run_{run_number}/preds_run{run_number}.parquet"
    if not preds_file.exists():
        return None
    
    df = pl.read_parquet(preds_file)
    
    # ⚠️ IMPORTANT: Les prédictions existantes ont été faites avec l'ancienne convention
    # où LABEL_REAL=0 et LABEL_SIMULATED=1, mais prob_real prenait probs[:, 1]
    # Donc prob_real représentait en fait P(SIMULATED), pas P(REAL) !
    # On doit inverser les prédictions pour l'analyse
    print(f"⚠️  Les prédictions du RUN {run_number} ont été faites avec l'ancienne convention")
    print(f"   Inversion de prob_real: prob_real → (1 - prob_real)")
    
    df = df.with_columns([
        (1.0 - pl.col("prob_real")).alias("prob_real")
    ])
    
    # Ajouter labels (nouvelle convention: REAL=1, SIMULATED=0)
    real_dir = RESULTS_DIR / "run_2_real"
    real_files = set()
    if real_dir.exists():
        real_files = {f.name for f in real_dir.glob("*.fasta")}
    
    def get_label(filename):
        return 1 if filename in real_files else 0
    
    df = df.with_columns([
        pl.col("filename").map_elements(get_label, return_dtype=pl.Int64).alias("true_label")
    ])
    
    return df


def plot_roc_with_thresholds(run_number):
    """Génère un plot ROC avec les différents seuils"""
    
    # Charger les prédictions
    df = load_predictions_with_labels(run_number)
    if df is None:
        print(f"⚠️ Pas de données pour RUN {run_number}")
        return
    
    # Calculer la courbe ROC
    y_true = df["true_label"].to_numpy()
    y_score = df["prob_real"].to_numpy()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    # Calculer le seuil optimal (Youden)
    optimal_threshold, j_score, optimal_idx = calculate_optimal_threshold_youden(fpr, tpr, thresholds)
    
    # Trouver l'index pour le seuil 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    
    # L'ancien seuil était l'AUC lui-même (ce qui n'a pas de sens)
    old_threshold = auc
    
    # Créer le plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Courbe ROC
    ax1.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    # Marquer les différents seuils
    ax1.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12, 
             label=f'Optimal (Youden): {optimal_threshold:.4f}')
    ax1.plot(fpr[idx_05], tpr[idx_05], 'go', markersize=12, 
             label=f'Threshold 0.5')
    
    # Note: l'ancien seuil (AUC comme seuil) n'a pas de sens sur la courbe ROC
    
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax1.set_title(f'RUN {run_number} - Courbe ROC avec seuils', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # Plot 2: J de Youden vs Threshold
    j_scores = tpr - fpr
    ax2.plot(thresholds, j_scores, linewidth=2, color='purple', label="Youden's J (TPR - FPR)")
    ax2.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Max J at {optimal_threshold:.4f}')
    ax2.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold 0.5')
    ax2.axvline(old_threshold, color='orange', linestyle='--', linewidth=2, 
                label=f'Ancien seuil (AUC={old_threshold:.4f})')
    
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel("Youden's J statistic", fontsize=12)
    ax2.set_title(f'RUN {run_number} - Statistique J de Youden', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"optimal_threshold_comparison_run{run_number}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualisation sauvegardée : {output_path}")
    plt.close()
    
    # Afficher les statistiques
    print(f"\n{'─'*80}")
    print(f"  RUN {run_number} - Comparaison des seuils")
    print(f"{'─'*80}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Ancien seuil (AUC comme seuil): {old_threshold:.4f}")
    print(f"  Nouveau seuil (Youden's J):     {optimal_threshold:.4f}")
    print(f"  Seuil 0.5 (référence):          0.5000")
    print(f"\n  Au seuil optimal (Youden):")
    print(f"    • TPR (Recall):  {tpr[optimal_idx]:.4f}")
    print(f"    • FPR:           {fpr[optimal_idx]:.4f}")
    print(f"    • J statistic:   {j_score:.4f}")
    print(f"\n  Au seuil 0.5:")
    print(f"    • TPR (Recall):  {tpr[idx_05]:.4f}")
    print(f"    • FPR:           {fpr[idx_05]:.4f}")
    print(f"    • J statistic:   {j_scores[idx_05]:.4f}")
    print(f"{'─'*80}\n")


def main():
    print("\n" + "="*80)
    print("VISUALISATION DU SEUIL OPTIMAL (YOUDEN'S J)")
    print("="*80 + "\n")
    
    print("📊 Génération des visualisations...\n")
    
    for run_number in [1, 2]:
        plot_roc_with_thresholds(run_number)
    
    print("\n" + "="*80)
    print("✅ VISUALISATION TERMINÉE")
    print(f"📂 Résultats disponibles dans : {OUTPUT_DIR}")
    print("="*80)
    print("\n📝 Explication du changement:")
    print("  • AVANT: Le pipeline utilisait l'AUC (ex: 0.87) comme seuil de classification")
    print("           → Cela n'a aucun sens statistique !")
    print("  • APRÈS: Le pipeline utilise le J de Youden (TPR - FPR) pour trouver")
    print("           le seuil optimal sur la courbe ROC")
    print("           → Maximise la séparation entre les deux classes")
    print("\n")


if __name__ == "__main__":
    main()

