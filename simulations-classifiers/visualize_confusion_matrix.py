import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re
import argparse

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

def plot_confusion_matrices(classif_dir, output_dir):
    """Plot confusion matrices for a specific group"""
    # Get all classifier types for this group
    classifier_types = get_classifier_types()
    
    n_classifiers = len(classifier_types)
        
    # Calculate subplot layout
    n_cols = min(4, n_classifiers)
    n_rows = n_classifiers // n_cols
    
    plt.figure(figsize=(20, 5 * n_rows))
        
    for classifier_idx, classifier in enumerate(classifier_types):
      idx = classifier_idx + 1
      plot_single_confusion_matrix(classif_dir, classifier, n_rows, n_cols, idx)
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved confusion matrix to {output_path}')


def plot_single_confusion_matrix(base_dir, classifier, n_rows, n_cols, idx):
    """Plot a single confusion matrix"""
    preds_path = os.path.join(base_dir, classifier, 'best_preds.parquet')
    
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
                
                plt.title(f'{classifier}\nAcc: {accuracy:.3f}', fontsize=8)
                plt.xlabel('Predicted', fontsize=8)
                plt.ylabel('True', fontsize=8)
                
                print(f'Created confusion matrix for {classifier}')
        except Exception as e:
            print(f'Error processing {preds_path}: {str(e)}')
    else:
        print(f'File not found: {preds_path}')

def create_all_confusion_matrices(base_dir, output_dir):
    """Create confusion matrices for all groups"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, group_info in groups.items():
        print(f"\nProcessing {group_info['name']}...")
        plot_confusion_matrix_group(base_dir, output_dir, group_name, group_info)


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrices for different classifiers')
    parser.add_argument('--classif', type=str, default="",
                      help='Path to the classifiers output directory')
    parser.add_argument('--output', type=str, default="",
                      help='Path to the confusion plot directory')
    args = parser.parse_args()
    
    # Create confusion matrices
    plot_confusion_matrices(args.classif, args.output)

if __name__ == "__main__":
    main()
