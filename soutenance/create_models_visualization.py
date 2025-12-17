"""
Script pour créer une visualisation attrayante des 4 meilleurs modèles
pour une présentation. Se concentre sur les modèles eux-mêmes (architecture, paramètres).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import json
from pathlib import Path

# Configuration des couleurs pour chaque modèle
MODEL_COLORS = {
    "AACnnClassifier": "#4ECDC4",  # Turquoise
    "DenseMsaClassifier": "#96CEB4",  # Vert clair
    "DenseSiteClassifier": "#FFEAA7",  # Jaune
    "LogisticRegressionClassifier": "#FF6B6B",  # Rouge
}

# Données des modèles (basées sur les résultats réels)
MODELS_DATA = {
    "AACnnClassifier": {
        "type": "CNN",
        "architecture": "Conv1D - ReLU - AdaptiveAvgPool - Dropout(0.2) - Linear",
        "layers": ["Conv1d(21-64)", "ReLU", "AdaptiveAvgPool1d", "Dropout(0.2)", "Linear(64-1)"],
        "n_params": 1537,
        "data_type": "SiteCompositionData",
        "input_shape": "(batch, sites, 21)",
        "f1_score": 0.838,
        "val_acc": 0.843,
        "best_epoch": 363,
        "training_time": 393,
        "hyperparams": {
            "batch_size": 64,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            "kernel_size": 1,
            "dropout": 0.2
        }
    },
    "DenseMsaClassifier": {
        "type": "Dense Neural Network",
        "architecture": "Linear - BatchNorm - ReLU - Dropout - Linear - BatchNorm - ReLU - Dropout - Linear",
        "layers": ["Linear(21-128)", "BatchNorm", "ReLU", "Dropout(0.4)", 
                  "Linear(128-64)", "BatchNorm", "ReLU", "Dropout(0.3)", "Linear(64-1)"],
        "n_params": 11649,
        "data_type": "MsaCompositionData",
        "input_shape": "(batch, 21)",
        "f1_score": 0.537,
        "val_acc": 0.640,
        "best_epoch": 8,
        "training_time": 16,
        "hyperparams": {
            "batch_size": 64,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            "dropout_1": 0.4,
            "dropout_2": 0.3
        }
    },
    "DenseSiteClassifier": {
        "type": "Dense Neural Network",
        "architecture": "Flatten - Linear - BatchNorm - ReLU - Dropout - Linear - BatchNorm - ReLU - Dropout - Linear",
        "layers": ["Flatten(sites×21)", "Linear(-128)", "BatchNorm", "ReLU", "Dropout(0.4)",
                  "Linear(128-64)", "BatchNorm", "ReLU", "Dropout(0.3)", "Linear(64-1)"],
        "n_params": 5153665,
        "data_type": "SiteCompositionData",
        "input_shape": "(batch, sites, 21)",
        "f1_score": 0.0,
        "val_acc": 0.529,
        "best_epoch": 1,
        "training_time": 26,
        "hyperparams": {
            "batch_size": 64,
            "max_epochs": 500,
            "early_stopping_patience": 20,
            "dropout_1": 0.4,
            "dropout_2": 0.3
        }
    },
    "LogisticRegressionClassifier": {
        "type": "Logistic Regression",
        "architecture": "Linear Classifier",
        "layers": ["Linear(21-1)", "Sigmoid"],
        "n_params": 22,  # 21 features + 1 bias
        "data_type": "MsaCompositionData",
        "input_shape": "(batch, 21)",
        "f1_score": 0.410,  # Moyenne des CV folds
        "val_acc": 0.380,  # Moyenne des CV folds
        "best_epoch": None,
        "training_time": 1,
        "hyperparams": {
            "cv": 50,
            "scale_features": True,
            "shuffle_data": True
        }
    }
}

def format_number(num):
    """Formate un nombre pour l'affichage"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(int(num))

def create_models_visualization(output_path="soutenance/models_comparison.png"):
    """Crée une visualisation attrayante des 4 modèles"""
    
    # Créer la figure avec une grille 2x2
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3, 
                         left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Titre principal
    fig.suptitle("Architecture et Paramètres des 4 Modèles de Classification", 
                 fontsize=28, fontweight='bold', y=0.98)
    
    model_names = list(MODELS_DATA.keys())
    
    # Créer un sous-graphique pour chaque modèle
    for idx, model_name in enumerate(model_names):
        row = idx // 2
        col = (idx % 2) * 2
        
        model_data = MODELS_DATA[model_name]
        color = MODEL_COLORS[model_name]
        
        # Zone principale du modèle (2 colonnes)
        ax_main = fig.add_subplot(gs[row, col:col+2])
        ax_main.axis('off')
        
        # Créer un cadre arrondi pour chaque modèle
        bbox = FancyBboxPatch((0, 0), 1, 1, 
                              boxstyle="round,pad=0.02", 
                              transform=ax_main.transAxes,
                              facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=3)
        ax_main.add_patch(bbox)
        
        # Titre du modèle
        ax_main.text(0.5, 0.95, model_name, 
                    ha='center', va='top', fontsize=20, fontweight='bold',
                    transform=ax_main.transAxes, color=color)
        
        # Type de modèle
        ax_main.text(0.5, 0.88, model_data["type"], 
                    ha='center', va='top', fontsize=14, style='italic',
                    transform=ax_main.transAxes, color='gray')
        
        # Architecture - Visualisation schématique
        y_start = 0.75
        y_spacing = 0.08
        
        # Titre section
        ax_main.text(0.05, y_start, "Architecture:", 
                    ha='left', va='top', fontsize=12, fontweight='bold',
                    transform=ax_main.transAxes)
        
        # Dessiner les couches
        layers = model_data["layers"]
        for i, layer in enumerate(layers):
            y_pos = y_start - 0.05 - (i * y_spacing)
            
            # Rectangle pour la couche
            rect = Rectangle((0.1, y_pos - 0.02), 0.8, 0.04,
                           transform=ax_main.transAxes,
                           facecolor=color, alpha=0.3,
                           edgecolor=color, linewidth=1.5)
            ax_main.add_patch(rect)
            
            # Texte de la couche
            ax_main.text(0.5, y_pos, layer, 
                        ha='center', va='center', fontsize=10,
                        transform=ax_main.transAxes, fontweight='bold')
            
            # Tiret entre les couches
            if i < len(layers) - 1:
                # Tiret horizontal simple
                mid_y = (y_pos - 0.02 + y_pos - y_spacing + 0.02) / 2
                dash = plt.Line2D([0.2, 0.8], [mid_y, mid_y],
                                 transform=ax_main.transAxes,
                                 color=color, linewidth=2, linestyle='--')
                ax_main.add_line(dash)
        
        # Paramètres et métriques (en bas)
        y_bottom = 0.15
        
        # Nombre de paramètres
        ax_main.text(0.05, y_bottom + 0.12, f"Paramètres: {format_number(model_data['n_params'])}", 
                    ha='left', va='top', fontsize=11, fontweight='bold',
                    transform=ax_main.transAxes)
        
        # Type de données
        ax_main.text(0.05, y_bottom + 0.08, f"Données: {model_data['data_type']}", 
                    ha='left', va='top', fontsize=10,
                    transform=ax_main.transAxes, color='gray')
        
        # Input shape
        ax_main.text(0.05, y_bottom + 0.04, f"Input: {model_data['input_shape']}", 
                    ha='left', va='top', fontsize=10,
                    transform=ax_main.transAxes, color='gray')
        
        # Performance (à droite)
        ax_main.text(0.55, y_bottom + 0.12, f"F1 Score: {model_data['f1_score']:.3f}", 
                    ha='left', va='top', fontsize=11, fontweight='bold',
                    transform=ax_main.transAxes, color='darkgreen')
        
        ax_main.text(0.55, y_bottom + 0.08, f"Val Accuracy: {model_data['val_acc']:.3f}", 
                    ha='left', va='top', fontsize=10,
                    transform=ax_main.transAxes, color='darkblue')
        
        if model_data['best_epoch']:
            ax_main.text(0.55, y_bottom + 0.04, f"Best Epoch: {model_data['best_epoch']}", 
                        ha='left', va='top', fontsize=10,
                        transform=ax_main.transAxes, color='gray')
        
        # Hyperparamètres (en bas, centré)
        hyperparams_items = list(model_data['hyperparams'].items())
        # Diviser en deux lignes si nécessaire
        if len(hyperparams_items) > 3:
            mid = len(hyperparams_items) // 2
            line1 = " - ".join([f"{k}={v}" for k, v in hyperparams_items[:mid]])
            line2 = " - ".join([f"{k}={v}" for k, v in hyperparams_items[mid:]])
            ax_main.text(0.5, y_bottom - 0.02, f"Hyperparamètres:\n{line1}\n{line2}", 
                        ha='center', va='top', fontsize=8,
                        transform=ax_main.transAxes, color='darkgray',
                        style='italic')
        else:
            hyperparams_text = " - ".join([f"{k}={v}" for k, v in hyperparams_items])
            ax_main.text(0.5, y_bottom - 0.02, f"Hyperparamètres: {hyperparams_text}", 
                        ha='center', va='top', fontsize=8,
                        transform=ax_main.transAxes, color='darkgray',
                        style='italic')
    
    # Graphique de comparaison des performances (en bas, sur toute la largeur)
    # Diviser en deux sous-graphiques: performances et paramètres
    ax_perf = fig.add_subplot(gs[2, :2])
    ax_params = fig.add_subplot(gs[2, 2:])
    
    # Graphique des performances
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    f1_scores = [MODELS_DATA[m]["f1_score"] for m in model_names]
    val_accs = [MODELS_DATA[m]["val_acc"] for m in model_names]
    colors_list = [MODEL_COLORS[m] for m in model_names]
    
    bars1 = ax_perf.bar(x_pos - width/2, f1_scores, width, 
                       label='F1 Score', color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax_perf.bar(x_pos + width/2, val_accs, width, 
                       label='Validation Accuracy', color=colors_list, alpha=0.5, 
                       edgecolor='black', linewidth=1.5, hatch='///')
    
    # Ajouter les valeurs sur les barres
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax_perf.set_xlabel('Modèles', fontsize=14, fontweight='bold')
    ax_perf.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax_perf.set_xticks(x_pos)
    ax_perf.set_xticklabels([m.replace('Classifier', '') for m in model_names], 
                           fontsize=11, rotation=0)
    ax_perf.set_ylim([0, 1.0])
    ax_perf.legend(loc='upper right', fontsize=12)
    ax_perf.grid(axis='y', alpha=0.3, linestyle='--')
    ax_perf.set_title('Comparaison des Performances', fontsize=16, fontweight='bold', pad=15)
    
    # Graphique du nombre de paramètres (échelle logarithmique)
    n_params_list = [MODELS_DATA[m]["n_params"] for m in model_names]
    bars_params = ax_params.bar(x_pos, n_params_list, width=0.6, 
                               color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax_params.set_yscale('log')
    ax_params.set_xlabel('Modèles', fontsize=14, fontweight='bold')
    ax_params.set_ylabel('Nombre de Paramètres (échelle log)', fontsize=14, fontweight='bold')
    ax_params.set_xticks(x_pos)
    ax_params.set_xticklabels([m.replace('Classifier', '') for m in model_names], 
                             fontsize=11, rotation=0)
    ax_params.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    ax_params.set_title('Complexité des Modèles', fontsize=16, fontweight='bold', pad=15)
    
    # Ajouter les valeurs
    for bar, n_params in zip(bars_params, n_params_list):
        height = bar.get_height()
        ax_params.text(bar.get_x() + bar.get_width()/2., height,
                      format_number(n_params),
                      ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Sauvegarder
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualisation sauvegardée: {output_path}")
    
    return fig

if __name__ == "__main__":
    # Créer la visualisation
    fig = create_models_visualization("soutenance/models_comparison.png")
    plt.close(fig)
    print("✓ Visualisation créée avec succès!")

