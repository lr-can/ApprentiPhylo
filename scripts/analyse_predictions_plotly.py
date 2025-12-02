#!/usr/bin/env python3
"""
Analyse des prédictions avec visualisations Plotly interactives
Affiche le seuil optimal de Youden calculé pour chaque run
"""

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

# Configuration
RESULTS_DIR = Path("results/classification")
OUTPUT_DIR = RESULTS_DIR / "predictions_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)


def calculate_youden_threshold(y_true, y_score):
    """Calcule le seuil optimal avec le J de Youden"""
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        auc = roc_auc_score(y_true, y_score)
        
        # Calculer le J de Youden pour chaque point
        j_scores = tpr - fpr
        
        # Filtrer les thresholds invalides (inf, nan, < 0, > 1)
        valid_mask = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
        
        if not np.any(valid_mask):
            # Aucun threshold valide, utiliser 0.5
            print(f"⚠️  Aucun threshold valide trouvé, utilisation de 0.5 par défaut")
            return {
                'threshold': 0.5,
                'tpr': 0.5,
                'fpr': 0.5,
                'j_score': 0.0,
                'auc': float(auc)
            }
        
        # Trouver le meilleur J parmi les thresholds valides
        valid_j_scores = j_scores[valid_mask]
        valid_thresholds = thresholds[valid_mask]
        valid_tpr = tpr[valid_mask]
        valid_fpr = fpr[valid_mask]
        
        optimal_idx_in_valid = np.argmax(valid_j_scores)
        
        optimal_threshold = valid_thresholds[optimal_idx_in_valid]
        optimal_tpr = valid_tpr[optimal_idx_in_valid]
        optimal_fpr = valid_fpr[optimal_idx_in_valid]
        optimal_j = valid_j_scores[optimal_idx_in_valid]
        
        return {
            'threshold': float(optimal_threshold),
            'tpr': float(optimal_tpr),
            'fpr': float(optimal_fpr),
            'j_score': float(optimal_j),
            'auc': float(auc)
        }
    except Exception as e:
        print(f"⚠️  Erreur dans calculate_youden_threshold: {e}")
        return {
            'threshold': 0.5,
            'tpr': 0.0,
            'fpr': 0.0,
            'j_score': 0.0,
            'auc': 0.5
        }


def load_predictions_with_labels(run_number):
    """Charge les prédictions et ajoute les vrais labels"""
    preds_file = RESULTS_DIR / f"run_{run_number}/preds_run{run_number}.parquet"
    if not preds_file.exists():
        return None, None
    
    df = pl.read_parquet(preds_file)
    
    # Note: Les données ont été régénérées avec la nouvelle convention des labels
    # (LABEL_SIMULATED=0, LABEL_REAL=1). Plus besoin d'inversion.
    print(f"✓ Chargement RUN {run_number} (convention: REAL=1, SIM=0)")
    
    # Ajouter labels (nouvelle convention: REAL=1, SIMULATED=0)
    # Pour RUN 1: utiliser les sources originales
    # Pour RUN 2: utiliser run_2_real et run_2_sim
    if run_number == 1:
        real_dir = Path("results/preprocessed/clean_data")
        sim_dir = Path("results/simulations")
    else:
        real_dir = RESULTS_DIR / "run_2_real"
        sim_dir = RESULTS_DIR / "run_2_sim"
    
    real_files = set()
    sim_files = set()
    
    if real_dir.exists():
        real_files = {f.name for f in real_dir.glob("*.fasta")}
    if sim_dir.exists():
        sim_files = {f.name for f in sim_dir.glob("*.fasta")}
    
    print(f"   Fichiers réels trouvés : {len(real_files)}")
    print(f"   Fichiers simulés trouvés : {len(sim_files)}")
    
    def get_label(filename):
        if filename in real_files:
            return 1  # REAL
        elif filename in sim_files:
            return 0  # SIMULATED
        # Les fichiers simulés peuvent avoir le suffixe "_sim"
        elif filename.endswith('_sim'):
            base_name = filename[:-4]  # Enlever "_sim"
            if base_name in sim_files or base_name in real_files:
                return 0  # SIMULATED (même si base est dans real, c'est une simulation)
            return 0  # Par défaut, assume simulated
        else:
            return -1  # Unknown (sera filtré)
    
    df = df.with_columns([
        pl.col("filename").map_elements(get_label, return_dtype=pl.Int64).alias("true_label")
    ])
    
    # Filtrer les lignes avec label inconnu
    n_before = len(df)
    df = df.filter(pl.col("true_label") != -1)
    n_after = len(df)
    if n_before != n_after:
        print(f"   ⚠️  {n_before - n_after} fichiers sans label ignorés")
    
    # Calculer le seuil optimal de Youden
    y_true = df["true_label"].to_numpy()
    y_score = df["prob_real"].to_numpy()
    youden_stats = calculate_youden_threshold(y_true, y_score)
    
    return df, youden_stats


def create_violin_and_box_plots():
    """Génère des violin plots transparents avec box plots intégrés"""
    print("\n📊 Génération des violin plots avec box plots intégrés...\n")
    
    # Charger les données
    run1_df, youden1 = load_predictions_with_labels(1)
    run2_df, youden2 = load_predictions_with_labels(2)
    
    # Créer subplots simples: 1 ligne, 2 colonnes (RUN 1 et RUN 2)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'RUN 1 - Seuil Youden: {youden1["threshold"]:.3f}',
            f'RUN 2 - Seuil Youden: {youden2["threshold"]:.3f}'
        ),
        horizontal_spacing=0.15
    )
    
    colors = {'Simulé': '#ff7f0e', 'Réel': '#2ca02c'}
    
    # ============= RUN 1 =============
    if run1_df is not None:
        for label, label_name in [(0, 'Simulé'), (1, 'Réel')]:
            subset = run1_df.filter(pl.col("true_label") == label)
            probs = subset["prob_real"].to_numpy()
            
            fig.add_trace(
                go.Violin(
                    y=probs,
                    name=f'{label_name} (n={len(probs)})',
                    legendgroup='run1',
                    scalegroup='run1',
                    x0=label_name,
                    line_color=colors[label_name],
                    fillcolor=colors[label_name],
                    opacity=0.3,  # Plus transparent pour voir le boxplot
                    meanline_visible=False,  # Pas de ligne de moyenne
                    box_visible=True,
                    showlegend=(label == 0),
                    points=False
                ),
                row=1, col=1
            )
        
        # Lignes de seuil pour RUN 1
        fig.add_hline(
            y=youden1['threshold'],
            line_dash="dash",
            line_color="red",
            line_width=2.5,
            annotation_text=f"Youden: {youden1['threshold']:.3f}",
            annotation_position="right",
            row=1, col=1
        )
        fig.add_hline(
            y=0.5,
            line_dash="dot",
            line_color="gray",
            line_width=1.5,
            opacity=0.6,
            annotation_text="0.5",
            annotation_position="left",
            row=1, col=1
        )
    
    # ============= RUN 2 =============
    if run2_df is not None:
        for label, label_name in [(0, 'Simulé'), (1, 'Réel')]:
            subset = run2_df.filter(pl.col("true_label") == label)
            probs = subset["prob_real"].to_numpy()
            
            fig.add_trace(
                go.Violin(
                    y=probs,
                    name=f'{label_name} (n={len(probs)})',
                    legendgroup='run2',
                    scalegroup='run2',
                    x0=label_name,
                    line_color=colors[label_name],
                    fillcolor=colors[label_name],
                    opacity=0.3,  # Plus transparent pour voir le boxplot
                    meanline_visible=False,  # Pas de ligne de moyenne
                    box_visible=True,
                    showlegend=False,
                    points=False
                ),
                row=1, col=2
            )
        
        # Lignes de seuil pour RUN 2
        fig.add_hline(
            y=youden2['threshold'],
            line_dash="dash",
            line_color="red",
            line_width=2.5,
            annotation_text=f"Youden: {youden2['threshold']:.3f}",
            annotation_position="right",
            row=1, col=2
        )
        fig.add_hline(
            y=0.5,
            line_dash="dot",
            line_color="gray",
            line_width=1.5,
            opacity=0.6,
            annotation_text="0.5",
            annotation_position="left",
            row=1, col=2
        )
    
    # Mise en forme des axes
    fig.update_yaxes(title_text="Probabilité prédite (classe RÉEL)", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Probabilité prédite (classe RÉEL)", range=[-0.05, 1.05], row=1, col=2)
    fig.update_xaxes(title_text="Type d'alignement", row=1, col=1)
    fig.update_xaxes(title_text="Type d'alignement", row=1, col=2)
    
    fig.update_layout(
        title_text="Distribution des probabilités - Violin Plots avec Box Plots intégrés",
        title_font_size=18,
        height=600,
        width=1400,
        showlegend=True,
        template="plotly_white"
    )
    
    # Export HTML et PNG
    html_path = OUTPUT_DIR / "violin_plots_with_boxes.html"
    png_path = OUTPUT_DIR / "violin_plots_with_boxes.png"
    
    fig.write_html(html_path)
    fig.write_image(png_path, width=1400, height=600, scale=2)
    
    print(f"✓ Violin plots (avec box plots intégrés) sauvegardés :")
    print(f"  - HTML (interactif) : {html_path}")
    print(f"  - PNG (statique)    : {png_path}")
    
    return youden1, youden2


def create_histograms(youden1, youden2):
    """Génère des histogrammes avec Plotly"""
    print("\n📊 Génération des histogrammes avec Plotly...\n")
    
    # Charger les données
    run1_df, _ = load_predictions_with_labels(1)
    run2_df, _ = load_predictions_with_labels(2)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'RUN 1 - Distribution (Seuil Youden: {youden1["threshold"]:.3f})',
            f'RUN 2 - Distribution (Seuil Youden: {youden2["threshold"]:.3f})'
        ),
        vertical_spacing=0.12
    )
    
    # RUN 1
    if run1_df is not None:
        sim = run1_df.filter(pl.col("true_label") == 0)["prob_real"].to_numpy()
        real = run1_df.filter(pl.col("true_label") == 1)["prob_real"].to_numpy()
        
        fig.add_trace(
            go.Histogram(
                x=sim,
                name=f'Simulé (n={len(sim)})',
                marker_color='#ff7f0e',
                opacity=0.6,
                nbinsx=50,
                legendgroup='run1'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=real,
                name=f'Réel (n={len(real)})',
                marker_color='#2ca02c',
                opacity=0.6,
                nbinsx=50,
                legendgroup='run1'
            ),
            row=1, col=1
        )
        
        # Ligne de seuil Youden
        fig.add_vline(
            x=youden1['threshold'],
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Youden: {youden1['threshold']:.3f}",
            annotation_position="top",
            row=1, col=1
        )
        
        # Ligne de référence 0.5
        fig.add_vline(
            x=0.5,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            opacity=0.5,
            row=1, col=1
        )
    
    # RUN 2
    if run2_df is not None:
        sim = run2_df.filter(pl.col("true_label") == 0)["prob_real"].to_numpy()
        real = run2_df.filter(pl.col("true_label") == 1)["prob_real"].to_numpy()
        
        fig.add_trace(
            go.Histogram(
                x=sim,
                name=f'Simulé (n={len(sim)})',
                marker_color='#ff7f0e',
                opacity=0.6,
                nbinsx=50,
                legendgroup='run2',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=real,
                name=f'Réel (n={len(real)})',
                marker_color='#2ca02c',
                opacity=0.6,
                nbinsx=50,
                legendgroup='run2',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Ligne de seuil Youden
        fig.add_vline(
            x=youden2['threshold'],
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Youden: {youden2['threshold']:.3f}",
            annotation_position="top",
            row=2, col=1
        )
        
        # Ligne de référence 0.5
        fig.add_vline(
            x=0.5,
            line_dash="dot",
            line_color="gray",
            line_width=1,
            opacity=0.5,
            row=2, col=1
        )
    
    # Mise en forme
    fig.update_xaxes(title_text="Probabilité prédite (classe RÉEL)", row=1, col=1)
    fig.update_xaxes(title_text="Probabilité prédite (classe RÉEL)", row=2, col=1)
    fig.update_yaxes(title_text="Fréquence", row=1, col=1)
    fig.update_yaxes(title_text="Fréquence", row=2, col=1)
    
    fig.update_layout(
        title_text="Histogrammes des probabilités prédites",
        title_font_size=18,
        height=900,
        width=1400,
        showlegend=True,
        barmode='overlay',
        template="plotly_white"
    )
    
    # Export HTML et PNG
    html_path = OUTPUT_DIR / "histograms_interactive.html"
    png_path = OUTPUT_DIR / "histograms_interactive.png"
    
    fig.write_html(html_path)
    fig.write_image(png_path, width=1400, height=900, scale=2)
    
    print(f"✓ Histogrammes sauvegardés :")
    print(f"  - HTML (interactif) : {html_path}")
    print(f"  - PNG (statique)    : {png_path}")


def create_boxplots_comparison(youden1, youden2):
    """Génère un boxplot comparatif entre RUN 1 et RUN 2"""
    print("\n📊 Génération du boxplot comparatif avec Plotly...\n")
    
    # Charger les données
    run1_df, _ = load_predictions_with_labels(1)
    run2_df, _ = load_predictions_with_labels(2)
    
    fig = go.Figure()
    
    colors = {'Simulé': '#ff7f0e', 'Réel': '#2ca02c'}
    
    # Préparer les données
    data_list = []
    
    for run_num, (run_df, youden) in enumerate([(run1_df, youden1), (run2_df, youden2)], 1):
        if run_df is None:
            continue
        
        for label, label_name in [(0, 'Simulé'), (1, 'Réel')]:
            subset = run_df.filter(pl.col("true_label") == label)
            probs = subset["prob_real"].to_numpy()
            
            fig.add_trace(
                go.Box(
                    y=probs,
                    name=f'{label_name}',
                    x=[f'RUN {run_num}'] * len(probs),
                    marker_color=colors[label_name],
                    legendgroup=label_name,
                    showlegend=(run_num == 1)
                )
            )
    
    # Ajouter les seuils de Youden
    fig.add_hline(
        y=youden1['threshold'],
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"RUN 1 Youden: {youden1['threshold']:.3f}",
        annotation_position="left"
    )
    
    fig.add_hline(
        y=youden2['threshold'],
        line_dash="dash",
        line_color="darkred",
        line_width=2,
        annotation_text=f"RUN 2 Youden: {youden2['threshold']:.3f}",
        annotation_position="right"
    )
    
    # Ligne de référence 0.5
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="gray",
        line_width=1,
        opacity=0.5
    )
    
    fig.update_layout(
        title_text="Comparaison RUN 1 vs RUN 2 - Boxplots avec seuils de Youden",
        title_font_size=18,
        yaxis_title="Probabilité prédite (classe RÉEL)",
        xaxis_title="",
        height=700,
        width=1200,
        showlegend=True,
        template="plotly_white",
        yaxis_range=[-0.05, 1.05]
    )
    
    # Export HTML et PNG
    html_path = OUTPUT_DIR / "boxplots_comparison_interactive.html"
    png_path = OUTPUT_DIR / "boxplots_comparison_interactive.png"
    
    fig.write_html(html_path)
    fig.write_image(png_path, width=1200, height=700, scale=2)
    
    print(f"✓ Boxplots comparatifs sauvegardés :")
    print(f"  - HTML (interactif) : {html_path}")
    print(f"  - PNG (statique)    : {png_path}")


def print_statistics(youden1, youden2):
    """Affiche les statistiques des seuils de Youden"""
    print("\n" + "="*80)
    print("STATISTIQUES DES SEUILS DE YOUDEN")
    print("="*80 + "\n")
    
    print("📊 RUN 1")
    print(f"  • Seuil optimal (Youden's J) : {youden1['threshold']:.4f}")
    print(f"  • AUC                        : {youden1['auc']:.4f}")
    print(f"  • TPR (Sensibilité)          : {youden1['tpr']:.4f}")
    print(f"  • FPR                        : {youden1['fpr']:.4f}")
    print(f"  • J statistic                : {youden1['j_score']:.4f}")
    
    print(f"\n📊 RUN 2")
    print(f"  • Seuil optimal (Youden's J) : {youden2['threshold']:.4f}")
    print(f"  • AUC                        : {youden2['auc']:.4f}")
    print(f"  • TPR (Sensibilité)          : {youden2['tpr']:.4f}")
    print(f"  • FPR                        : {youden2['fpr']:.4f}")
    print(f"  • J statistic                : {youden2['j_score']:.4f}")
    
    print("\n" + "="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("ANALYSE DES PRÉDICTIONS AVEC PLOTLY")
    print("="*80 + "\n")
    
    # Créer les visualisations
    youden1, youden2 = create_violin_and_box_plots()
    create_histograms(youden1, youden2)
    create_boxplots_comparison(youden1, youden2)
    
    # Afficher les statistiques
    print_statistics(youden1, youden2)
    
    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE")
    print(f"📂 Résultats disponibles dans : {OUTPUT_DIR}")
    print("="*80)
    print("\n💡 Les fichiers HTML sont interactifs (zoom, hover, export)")
    print("💡 Les fichiers PNG sont de haute qualité pour les publications\n")


if __name__ == "__main__":
    main()

