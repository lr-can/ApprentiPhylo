import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re
import argparse

def get_simulation_groups():
    """Define simulation groupings"""
    return {
        'group1': {
            'name': 'Basic Models',
            'simulations': [
                'DSO78_F',
                'JTT92_F',
                'LG08_F',
                'WAG_F'
            ]
        },
        'group2': {
            'name': 'WAG Posterior Models',
            'subgroups': {
                'with_data2': [
                    'WAG_F_P_DATA2_E_0',
                    'WAG_F_P_DATA2_E_0.05',
                    'WAG_F_P_DATA2_E_0.1',
                    'WAG_F_P_DATA2_E_0.2',
                    'WAG_F_P_DATA2_E_0.5'
                ],
                'without_data2': [
                    'WAG_F_P_E_0',
                    'WAG_F_P_E_0.05',
                    'WAG_F_P_E_0.1',
                    'WAG_F_P_E_0.2',
                    'WAG_F_P_E_0.5'
                ]
            }
        },
        'group3': {
            'name': 'WAG Sampling Sequence Models',
            'subgroups': {
                'without_data2': [
                    'WAG_EP_R_0',
                    'WAG_EP_R_0.05',
                    'WAG_EP_R_0.1',
                    'WAG_EP_R_0.2',
                    'WAG_EP_R_0.5'
                ],
                'with_data2': [
                    'WAG_EP_DATA2_R_0',
                    'WAG_EP_DATA2_R_0.05',
                    'WAG_EP_DATA2_R_0.1',
                    'WAG_EP_DATA2_R_0.2',
                    'WAG_EP_DATA2_R_0.5'
                ]
            }
        },
        'group4': {
            'name': 'WAG Basic Comparison',
            'subgroups': {
                'without_data2': [
                    'WAG_F',
                    'WAG_F_P_E_0',
                    'WAG_EP_R_0',
                    'WAG_F_EP'
                ],
                'with_data2': [
                    'WAG_F_P_DATA2_E_0',
                    'WAG_EP_DATA2_R_0',
                    'WAG_F_EP_DATA2'
                ]
            }
        }
    }

def get_classifier_types():
    """Get all classifier types"""
    return [
        'AACnnClassifier',
        'DenseSiteClassifier',
        'DenseMsaClassifier',
        'LogisticRegressionClassifier'
    ]

def get_true_pred_columns(df):
    """Automatically get the column names for true and predicted values"""
    if 'target' in df.columns and 'pred' in df.columns:
        return 'target', 'pred'
    return None, None

def plot_confusion_matrix_group(base_dir, output_dir, group_name, group_info):
    """Plot confusion matrices for a specific group"""
    # Get all classifier types for this group
    classifier_types = get_classifier_types()
    
    if 'simulations' in group_info:
        # For groups without subgroups
        simulations = group_info['simulations']
        n_sims = len(simulations)
        n_classifiers = len(classifier_types)
        
        # Calculate subplot layout
        n_cols = min(4, n_classifiers)
        n_rows = n_sims
        
        plt.figure(figsize=(20, 5 * n_rows))
        
        for sim_idx, sim_type in enumerate(simulations):
            for classifier_idx, classifier in enumerate(classifier_types):
                idx = sim_idx * n_cols + classifier_idx + 1
                plot_single_confusion_matrix(base_dir, sim_type, classifier, n_rows, n_cols, idx)
        
        plt.tight_layout()
        output_path = os.path.join('results', 'simulations-classifiers', 'visualization', 'confusion_matrix', f'confusion_matrix_{group_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved confusion matrix to {output_path}')
    else:
        # For groups with subgroups
        all_simulations = []
        for subgroup_name, simulations in group_info['subgroups'].items():
            all_simulations.extend(simulations)
        
        n_sims = len(all_simulations)
        n_classifiers = len(classifier_types)
        
        # Calculate subplot layout
        n_cols = min(4, n_classifiers)
        n_rows = n_sims
        
        plt.figure(figsize=(20, 5 * n_rows))
        
        for sim_idx, sim_type in enumerate(all_simulations):
            for classifier_idx, classifier in enumerate(classifier_types):
                idx = sim_idx * n_cols + classifier_idx + 1
                plot_single_confusion_matrix(base_dir, sim_type, classifier, n_rows, n_cols, idx)
        
        plt.tight_layout()
        output_path = os.path.join('results', 'simulations-classifiers', 'visualization', 'confusion_matrix', f'confusion_matrix_{group_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved confusion matrix to {output_path}')

def plot_single_confusion_matrix(base_dir, sim_type, classifier, n_rows, n_cols, idx):
    """Plot a single confusion matrix"""
    preds_path = os.path.join(base_dir, sim_type, classifier, 'best_preds.parquet')
    
    if os.path.exists(preds_path):
        try:
            df = pd.read_parquet(preds_path)
            true_col, pred_col = get_true_pred_columns(df)
            
            if true_col and pred_col:
                y_true = df[true_col].astype(int)
                y_pred = df[pred_col].astype(int)
                
                plt.subplot(n_rows, n_cols, idx)
                
                cm = confusion_matrix(y_true, y_pred)
                cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                accuracy = np.sum(np.diag(cm)) / np.sum(cm)
                
                # Create annotation with percentage
                annot = np.array([[f'{val:.1f}%' for val in row] for row in cm_percentage])
                
                sns.heatmap(cm_percentage, annot=annot, fmt='', cmap='Blues',
                          xticklabels=['Simulated MSA', 'Empirical MSA'],
                          yticklabels=['Simulated MSA', 'Empirical MSA'])
                
                plt.title(f'{sim_type}\n{classifier}\nAcc: {accuracy:.3f}', fontsize=8)
                plt.xlabel('Predicted', fontsize=8)
                plt.ylabel('True', fontsize=8)
                
                print(f'Created confusion matrix for {sim_type} - {classifier}')
        except Exception as e:
            print(f'Error processing {preds_path}: {str(e)}')
    else:
        print(f'File not found: {preds_path}')

def create_all_confusion_matrices(base_dir, output_dir):
    """Create confusion matrices for all groups"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    groups = get_simulation_groups()
    for group_name, group_info in groups.items():
        print(f"\nProcessing {group_info['name']}...")
        plot_confusion_matrix_group(base_dir, output_dir, group_name, group_info)

def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrices for different simulation groups')
    parser.add_argument('--runs_path', type=str, default='runs_viridiplantae',
                      help='Path to the runs directory (default: runs_viridiplantae)')
    args = parser.parse_args()
    
    # Set output directory
    output_dir = os.path.join('results', 'simulations-classifiers', 'visualization', 'confusion_matrix')
    
    # Create confusion matrices
    create_all_confusion_matrices(args.runs_path, output_dir)

if __name__ == "__main__":
    main()
