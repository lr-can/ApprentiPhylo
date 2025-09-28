import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_log_file(log_path):
    """Parse training log file and extract training metrics"""
    epochs = []
    train_losses = []
    valid_losses = []
    accuracies = []
    f1_scores = []
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                # Match training log line, format: 2025-05-25 05:01:33       1        0.682        0.681       0.372       0.000   0.00000      *
                match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    valid_loss = float(match.group(3))
                    accuracy = float(match.group(4))
                    f1_score = float(match.group(5))
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                    accuracies.append(accuracy)
                    f1_scores.append(f1_score)
    except FileNotFoundError:
        print(f"Warning: Log file {log_path} not found")
        return None
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'accuracies': accuracies,
        'f1_scores': f1_scores
    }


def get_classifier_types():
    """Get classifier types to plot"""
    return [
        'AACnnClassifier',
        'DenseMsaClassifier',
        'DenseSiteClassifier'
    ]


def plot_metrics(classif_dir, output_dir):
    """Plot combined figures for all metrics"""
    classifiers = get_classifier_types()
    
    # Create a large figure with 4 rows (metrics) and 3 columns (classifiers)
    fig, axes = plt.subplots(4, 3, figsize=(24, 32))
    fig.suptitle('Training Metrics Across Classifiers', fontsize=24, y=1.02)
    
    # Set metric names and corresponding data keys
    metrics = [
        ('Training Loss', 'train_losses'),
        ('Validation Loss', 'valid_losses'),
        ('Accuracy', 'accuracies'),
        ('F1 Score', 'f1_scores')
    ]
    
    # Iterate over each metric and classifier
    for metric_idx, (metric_name, metric_key) in enumerate(metrics):
        for classifier_idx, classifier in enumerate(classifiers):
            ax = axes[metric_idx, classifier_idx]
            
            # Set subplot title
            ax.set_title(f'{classifier}\n{metric_name}', fontsize=16)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.grid(True)
            
            log_path = os.path.join(classif_dir, classifier, "training.log")
            data = parse_log_file(log_path)
            if data is None:
              continue
                            
            # # Get style
            # style = get_style_for_simulation(sim_type, has_data2)
                            
            # Plot data
            line = ax.plot(data['epochs'], data[metric_key], 
                           # color=style['color'],
                           # linestyle=style['linestyle'],
                           # marker=style['marker'],
                           markersize=6,
                           markevery=5,
                           linewidth=2,
                           label=classifier_idx)[0]
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    # Merge all labels, show legend only once
    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.87, 1), fontsize=12, title='Simulation Types')
    # Save figure
    output_path = os.path.join(output_dir, 'combined_metrics.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved combined metrics plot to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrices for different classifiers')
    parser.add_argument('--classif', type=str, default="",
                      help='Path to the classifiers output directory')
    parser.add_argument('--output', type=str, default="",
                      help='Path to the confusion plot directory')
    args = parser.parse_args()
    
    # Create confusion matrices
    plot_metrics(args.classif, args.output)

if __name__ == '__main__':
    main()
