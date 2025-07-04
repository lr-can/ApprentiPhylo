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

def get_simulation_groups():
    """Define simulation groupings"""
    return {
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

def get_classifiers():
    """Get classifier types to plot"""
    return [
        'AACnnClassifier',
        'DenseMsaClassifier',
        'DenseSiteClassifier'
    ]

def get_style_for_simulation(sim_type, has_data2):
    """Get style for simulation type, color consistent with comparison plot"""
    # Color scheme consistent with comparison plot
    color_map = {
        'DSO78_F': '#8c564b',  # brown
        'JTT92_F': '#1f77b4',  # blue
        'LG08_F': '#9467bd',   # purple
        'WAG_F': '#ff7f0e',    # orange
        'WAG_F_P_E_0': '#2ca02c',
        'WAG_F_P_E_0.05': '#2ca02c',
        'WAG_F_P_E_0.1': '#2ca02c',
        'WAG_F_P_E_0.2': '#2ca02c',
        'WAG_F_P_E_0.5': '#2ca02c',
        'WAG_EP_R_0': '#d62728',
        'WAG_EP_R_0.05': '#d62728',
        'WAG_EP_R_0.1': '#d62728',
        'WAG_EP_R_0.2': '#d62728',
        'WAG_EP_R_0.5': '#d62728',
        'WAG_F_EP': '#17becf',
        'WAG_F_P_DATA2_E_0': '#2ca02c',
        'WAG_F_P_DATA2_E_0.05': '#2ca02c',
        'WAG_F_P_DATA2_E_0.1': '#2ca02c',
        'WAG_F_P_DATA2_E_0.2': '#2ca02c',
        'WAG_F_P_DATA2_E_0.5': '#2ca02c',
        'WAG_EP_DATA2_R_0': '#d62728',
        'WAG_EP_DATA2_R_0.05': '#d62728',
        'WAG_EP_DATA2_R_0.1': '#d62728',
        'WAG_EP_DATA2_R_0.2': '#d62728',
        'WAG_EP_DATA2_R_0.5': '#d62728',
        'WAG_F_EP_DATA2': '#17becf',
    }
    marker_map = {
        'DSO78_F': 'o',
        'JTT92_F': 's',
        'LG08_F': '^',
        'WAG_F': 'D',
        'WAG_F_P_E_0': 'o',
        'WAG_F_P_E_0.05': 'o',
        'WAG_F_P_E_0.1': 'o',
        'WAG_F_P_E_0.2': 'o',
        'WAG_F_P_E_0.5': 'o',
        'WAG_EP_R_0': 's',
        'WAG_EP_R_0.05': 's',
        'WAG_EP_R_0.1': 's',
        'WAG_EP_R_0.2': 's',
        'WAG_EP_R_0.5': 's',
        'WAG_F_EP': '^',
        'WAG_F_P_DATA2_E_0': 'o',
        'WAG_F_P_DATA2_E_0.05': 'o',
        'WAG_F_P_DATA2_E_0.1': 'o',
        'WAG_F_P_DATA2_E_0.2': 'o',
        'WAG_F_P_DATA2_E_0.5': 'o',
        'WAG_EP_DATA2_R_0': 's',
        'WAG_EP_DATA2_R_0.05': 's',
        'WAG_EP_DATA2_R_0.1': 's',
        'WAG_EP_DATA2_R_0.2': 's',
        'WAG_EP_DATA2_R_0.5': 's',
        'WAG_F_EP_DATA2': '^',
    }
    color = color_map.get(sim_type, 'gray')
    marker = marker_map.get(sim_type, 'o')
    linestyle = '-' if has_data2 else '--'
    return {'color': color, 'marker': marker, 'linestyle': linestyle}

def plot_metrics(base_dir, output_dir):
    """Plot combined figures for all metrics"""
    groups = get_simulation_groups()
    classifiers = get_classifiers()
    
    # Create a large figure with 4 rows (metrics) and 3 columns (classifiers)
    fig, axes = plt.subplots(4, 3, figsize=(24, 32))
    fig.suptitle('Training Metrics Across Different Models and Classifiers', fontsize=24, y=1.02)
    
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
            
            # Iterate over each group
            for group_name, group_info in groups.items():
                if 'subgroups' in group_info:
                    for subgroup_name, simulations in group_info['subgroups'].items():
                        has_data2 = 'data2' in subgroup_name
                        for sim_type in simulations:
                            log_path = os.path.join(base_dir, sim_type, classifier, 'training.log')
                            data = parse_log_file(log_path)
                            if data is None:
                                continue
                            
                            # Get style
                            style = get_style_for_simulation(sim_type, has_data2)
                            
                            # Plot data
                            line = ax.plot(data['epochs'], data[metric_key], 
                                         color=style['color'],
                                         linestyle=style['linestyle'],
                                         marker=style['marker'],
                                         markersize=6,
                                         markevery=5,
                                         linewidth=2,
                                         label=f'{sim_type} ({subgroup_name})')[0]
    
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

def plot_metrics_for_densesite(base_dir, output_dir):
    """Plot all training metrics for DenseSiteClassifier, 2x2 layout, style consistent with main plot"""
    classifier = 'DenseSiteClassifier'
    groups = get_simulation_groups()
    metrics = [
        ('Training Loss', 'train_losses'),
        ('Validation Loss', 'valid_losses'),
        ('Accuracy', 'accuracies'),
        ('F1 Score', 'f1_scores')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DenseSiteClassifier Training Metrics Across Simulations', fontsize=20, y=1.02)
    legend_handles = []
    legend_labels = []
    for idx, (metric_name, metric_key) in enumerate(metrics):
        row, col = divmod(idx, 2)
        ax = axes[row, col]
        ax.set_title(metric_name, fontsize=16)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True)
        for group_name, group_info in groups.items():
            if 'subgroups' in group_info:
                for subgroup_name, simulations in group_info['subgroups'].items():
                    has_data2 = 'data2' in subgroup_name
                    for sim_type in simulations:
                        log_path = os.path.join(base_dir, sim_type, classifier, 'training.log')
                        data = parse_log_file(log_path)
                        if data is None:
                            continue
                        style = get_style_for_simulation(sim_type, has_data2)
                        line = ax.plot(data['epochs'], data[metric_key],
                                     color=style['color'],
                                     linestyle=style['linestyle'],
                                     marker=style['marker'],
                                     markersize=6,
                                     markevery=5,
                                     linewidth=2,
                                     label=f'{sim_type} ({subgroup_name})')[0]
                        if idx == 0:
                            legend_handles.append(line)
                            legend_labels.append(f'{sim_type} ({subgroup_name})')
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    fig.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(0.87, 1), fontsize=12, title='Simulation Types')
    output_path = os.path.join(output_dir, 'densesite_metrics.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved DenseSiteClassifier metrics plot to {output_path}')

def main():
    base_dir = "runs_viridiplantae"
    output_dir = os.path.join('results', 'simulations-classifiers', 'visualization', 'train_valid_plot')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_metrics(base_dir, output_dir)
    plot_metrics_for_densesite(base_dir, output_dir)

if __name__ == '__main__':
    main()
