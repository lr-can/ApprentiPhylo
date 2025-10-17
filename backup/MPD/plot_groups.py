import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plot_groups.log'),
    ]
)

def extract_number(model_name):
    """Extract number from model name"""
    match = re.search(r'([\d\.]+)$', model_name)
    return float(match.group(1)) if match else 0

def read_distances_from_csv(csv_file):
    """Read distance data from CSV file"""
    distances = []
    try:
        with open(csv_file) as f:
            header = f.readline().strip().split(',')
            if 'Average_Distance' not in header:
                logging.warning(f"Skip {csv_file}: No 'Average_Distance' column.")
                return []
            idx = header.index('Average_Distance')
            for line in f:
                parts = line.strip().split(',')
                if len(parts) <= idx:
                    continue
                val = parts[idx]
                if val not in ['NA', 'ERROR']:
                    try:
                        distances.append(float(val))
                    except ValueError:
                        continue
    except Exception as e:
        logging.error(f"Error reading {csv_file}: {str(e)}")
    return distances

def plot_distance_distribution(distances, output_path, title="Distance Distribution"):
    """Plot single distance distribution"""
    if not distances:
        logging.warning(f"No valid distances to plot for {title}")
        return

    distances = np.array(distances)
    # Remove outliers
    lower, upper = np.percentile(distances, [1, 99])
    filtered = distances[(distances >= lower) & (distances <= upper)]
    
    mean = np.mean(filtered)
    median = np.median(filtered)

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.histplot(filtered, bins=30, kde=True, color="#4C72B0", edgecolor="black", alpha=0.7)
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='-.', linewidth=2, label=f'Median: {median:.2f}')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.xlabel('Distance to Closest Empirical Sequence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.text(0.98, 0.95, f'N={len(filtered)}', ha='right', va='top', transform=plt.gca().transAxes, fontsize=11)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_combined_distributions(all_distances, output_path, title="Combined Distance Distributions"):
    """Plot combined distribution"""
    data = []
    for model, dists in all_distances.items():
        for v in dists:
            data.append({'Model': model, 'Distance': v})
    df = pd.DataFrame(data)
    
    # Remove outliers
    df = df.groupby('Model').apply(
        lambda g: g[g['Distance'].between(*np.percentile(g['Distance'], [1, 99]))]
    ).reset_index(drop=True)
    
    # Sort by number in model name
    unique_models = sorted(df['Model'].unique(), key=extract_number)
    df['Model'] = pd.Categorical(df['Model'], categories=unique_models, ordered=True)
    
    g = sns.FacetGrid(df, row="Model", hue="Model", aspect=4, height=1.2, sharex=True, sharey=False)
    g.map(sns.histplot, "Distance", bins=30, stat="count", element="step", fill=True, alpha=0.7)
    g.set_titles(row_template="{row_name}", size=9)
    g.set_xlabels("Distance to Closest Empirical Sequence", fontsize=12)
    g.set_ylabels("Frequency", fontsize=10)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for ax in g.axes.flat:
        ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)
    
    g.add_legend(title="", adjust_subtitles=True)
    for text in g._legend.texts:
        text.set_fontsize(8)
    g._legend.set_bbox_to_anchor((1.01, 0.5))
    g._legend.set_frame_on(False)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Combined plot saved to: {output_path}")

def plot_combined_boxplot(all_distances, output_path, title="Combined Distance Boxplot"):
    """Plot combined boxplot"""
    data = []
    for model, dists in all_distances.items():
        arr = np.array(dists)
        if len(arr) > 0:
            lower, upper = np.percentile(arr, [1, 99])
            arr = arr[(arr >= lower) & (arr <= upper)]
            for v in arr:
                data.append({'Model': model, 'Distance': v})
    df = pd.DataFrame(data)
    
    # Sort by number in model name
    unique_models = sorted(df['Model'].unique(), key=extract_number)
    df['Model'] = pd.Categorical(df['Model'], categories=unique_models, ordered=True)
    
    plt.figure(figsize=(max(8, len(df['Model'].unique())*1.5), 6))
    sns.set(style="whitegrid", font_scale=1.2)
    ax = sns.boxplot(x="Model", y="Distance", data=df, palette="Set2", width=0.6, fliersize=2, linewidth=1.5, showfliers=False)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Distance to Closest Empirical Sequence", fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Boxplot saved to: {output_path}")

def process_group_folder(group_folder, output_folder):
    """Process a single group folder"""
    all_distances = {}
    group_all_distances = {}  # Store all distance data for the entire group
    
    # Check if with_data2 and without_data2 subfolders exist
    has_data2_subfolders = False
    for subfolder in ['with_data2', 'without_data2']:
        if os.path.exists(os.path.join(group_folder, subfolder)):
            has_data2_subfolders = True
            break
    
    if has_data2_subfolders:
        # Process with_data2 and without_data2 subfolders
        for subfolder in ['with_data2', 'without_data2']:
            subfolder_path = os.path.join(group_folder, subfolder)
            if not os.path.exists(subfolder_path):
                continue
                
            subfolder_distances = {}
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.startswith('distance_results_') and file.endswith('.csv'):
                        csv_file = os.path.join(root, file)
                        model_name = os.path.basename(root)
                        distances = read_distances_from_csv(csv_file)
                        if distances:
                            subfolder_distances[model_name] = distances
                            # Add to group's total data
                            group_all_distances[f"{subfolder}_{model_name}"] = distances
            
            if subfolder_distances:
                # Create output directory for subfolder
                subfolder_output = os.path.join(output_folder, subfolder)
                os.makedirs(subfolder_output, exist_ok=True)
                
                # Plot combined distribution
                combined_plot_path = os.path.join(subfolder_output, 'combined_distributions.png')
                plot_combined_distributions(subfolder_distances, combined_plot_path, 
                                         f"Combined Distance Distributions - {subfolder}")
                
                boxplot_path = os.path.join(subfolder_output, 'combined_boxplot.png')
                plot_combined_boxplot(subfolder_distances, boxplot_path,
                                   f"Combined Distance Boxplot - {subfolder}")
    else:
        # Process normal folders
        for root, _, files in os.walk(group_folder):
            for file in files:
                if file.startswith('distance_results_') and file.endswith('.csv'):
                    csv_file = os.path.join(root, file)
                    model_name = os.path.basename(root)
                    distances = read_distances_from_csv(csv_file)
                    if distances:
                        all_distances[model_name] = distances
                        group_all_distances[model_name] = distances
    
        if all_distances:
            # Plot combined distribution
            combined_plot_path = os.path.join(output_folder, 'combined_distributions.png')
            plot_combined_distributions(all_distances, combined_plot_path)
            
            boxplot_path = os.path.join(output_folder, 'combined_boxplot.png')
            plot_combined_boxplot(all_distances, boxplot_path)
    
    # Generate combined plot for the entire group
    if group_all_distances:
        group_name = os.path.basename(group_folder)
        # Plot combined distribution for the entire group
        group_combined_plot_path = os.path.join(output_folder, f'{group_name}_all_combined_distributions.png')
        plot_combined_distributions(group_all_distances, group_combined_plot_path,
                                  f"Combined Distance Distributions - {group_name} ")
        
        # Plot combined boxplot for the entire group
        group_boxplot_path = os.path.join(output_folder, f'{group_name}_all_combined_boxplot.png')
        plot_combined_boxplot(group_all_distances, group_boxplot_path,
                            f"Combined Distance Boxplot - {group_name} ")

def main():
    """Main function"""
    # Set input and output directories
    input_folder = "results/MPD/viridiplantae_group_results"
    output_folder = "results/MPD/viridiplantae_group_plots"
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each group folder
    for group_folder in os.listdir(input_folder):
        group_path = os.path.join(input_folder, group_folder)
        if os.path.isdir(group_path):
            logging.info(f"Processing group folder: {group_folder}")
            group_output = os.path.join(output_folder, group_folder)
            process_group_folder(group_path, group_output)

if __name__ == "__main__":
    main() 