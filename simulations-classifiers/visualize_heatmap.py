import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

def extract_number_from_name(name):
    """Extract the numeric value from the simulation name"""
    match = re.search(r'E_(\d+\.?\d*)', name)
    if match:
        return float(match.group(1))
    return 0

def get_wag_simulation_types(base_dir):
    """Get WAG simulation types and sort them by their numeric values"""
    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Filter for WAG_F_P_E and WAG_F_P_DATA2_E
    wag_e_sims = [d for d in all_dirs if d.startswith('WAG_F_P_E_') and 'DATA2' not in d]
    wag_data2_e_sims = [d for d in all_dirs if d.startswith('WAG_F_P_DATA2_E_')]
    
    # Sort by the numeric value
    wag_e_sims.sort(key=extract_number_from_name)
    wag_data2_e_sims.sort(key=extract_number_from_name)
    
    return wag_e_sims, wag_data2_e_sims

def get_classifier_types():
    """Get all classifier types"""
    return [
        'AACnnClassifier',
        'DenseSiteClassifier',
        'DenseMsaClassifier',
        'LogisticRegressionClassifier'
    ]

def get_metric_value(base_dir, simulation_type, classifier_type, metric='accuracy'):
    """Get either accuracy or F1 score from summary.json"""
    summary_path = os.path.join(base_dir, simulation_type, classifier_type, 'summary.json')
    try:
        if not os.path.exists(summary_path):
            return None
            
        with open(summary_path, 'r') as f:
            data = json.load(f)
            if classifier_type == 'LogisticRegressionClassifier':
                if metric == 'accuracy':
                    key = 'fold_accuracies'
                else:  # f1_score
                    key = 'fold_f1_scores'
                if key in data:
                    values = [float(val) for val in data[key]]
                    return np.mean(values) if values else None
            else:
                key = 'val_acc' if metric == 'accuracy' else 'f1_score'
                return data.get(key, None)
    except Exception as e:
        print(f"Warning: Error processing {summary_path}: {str(e)}")
        return None

def create_combined_wag_heatmap(data, simulation_types, classifier_types, title, output_path):
    """Create a combined heatmap for WAG simulations"""
    # Convert to DataFrame
    df = pd.DataFrame(data, 
                     index=simulation_types,
                     columns=classifier_types)
    
    # Create output directory
    output_dir = os.path.join('results', 'simulations-classifiers', 'visualization', 'wag_heatmaps')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(output_path))
    
    # Create the heatmap
    plt.figure(figsize=(12, len(simulation_types) * 0.4 + 2))
    
    # Calculate vmin and vmax, excluding NaN values
    valid_data = df.values[~np.isnan(df.values)]
    if len(valid_data) > 0:
        vmin = np.min(valid_data)
        vmax = np.max(valid_data)
    else:
        vmin = 0
        vmax = 1
    
    # Create heatmap with custom settings
    sns.heatmap(df, 
                annot=True,
                cmap='YlOrRd',
                fmt='.3f',
                cbar_kws={'label': title},
                vmin=vmin,
                vmax=vmax)
    
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {title} heatmap to {output_path}')

def create_combined_wag_heatmaps(base_dir):
    """Create combined heatmaps for WAG_F_P_E and WAG_F_P_DATA2_E simulations"""
    # Get simulation types and classifiers
    wag_e_sims, wag_data2_e_sims = get_wag_simulation_types(base_dir)
    classifier_types = get_classifier_types()
    
    if not classifier_types:
        print("Error: No classifier types found")
        return
    
    # Create data matrices for both accuracy and F1 score
    metrics = ['accuracy', 'f1_score']
    for metric in metrics:
        # Combine all simulation types
        all_sim_types = []
        combined_data = []
        
        # Add WAG_F_P_E simulations
        for sim_type in wag_e_sims:
            row = []
            for classifier in classifier_types:
                value = get_metric_value(base_dir, sim_type, classifier, metric)
                row.append(value if value is not None else np.nan)
            combined_data.append(row)
            all_sim_types.append(sim_type)
        
        # Add WAG_F_P_DATA2_E simulations
        for sim_type in wag_data2_e_sims:
            row = []
            for classifier in classifier_types:
                value = get_metric_value(base_dir, sim_type, classifier, metric)
                row.append(value if value is not None else np.nan)
            combined_data.append(row)
            all_sim_types.append(sim_type)
        
        # Create combined heatmap
        metric_name = 'Accuracy' if metric == 'accuracy' else 'F1 Score'
        if combined_data:
            create_combined_wag_heatmap(
                combined_data,
                all_sim_types,
                classifier_types,
                f'Best {metric_name} for WAG_F_P_E and WAG_F_P_DATA2_E Simulations',
                f'combined_wag_{metric.lower()}_heatmap.png'
            )

if __name__ == "__main__":
    base_dir = "runs_viridiplantae"
    create_combined_wag_heatmaps(base_dir)
