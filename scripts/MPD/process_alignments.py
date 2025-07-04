import os
import subprocess
from Bio import SeqIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo import read
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys
import logging
import pandas as pd
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_alignments.log'),
    ]
)

def create_directories(simulation_type, output_dir='.'):
    """Create necessary directories for a given simulation type"""
    try:
        dirs = [
            os.path.join(output_dir, 'combine_aln', simulation_type),
            os.path.join(output_dir, 'combine_tree', simulation_type),
            os.path.join(output_dir, 'results', simulation_type),
            os.path.join(output_dir, 'plots', simulation_type)
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
    except Exception as e:
        logging.error(f"Error creating directories: {str(e)}")
        raise

def combine_alignments(empirical_file, simulation_file, output_file):
    """Combine two alignment files"""
    records = []
    for record in SeqIO.parse(empirical_file, "fasta"):
        record.id = f"empirical_{record.id}"
        records.append(record)
    for record in SeqIO.parse(simulation_file, "fasta"):
        record.id = f"simulation_{record.id}"
        records.append(record)
    SeqIO.write(records, output_file, "fasta")

def generate_tree(alignment_file, tree_file):
    """Generate phylogenetic tree using FastTree"""
    try:
        # Use quotes to wrap file paths and use subprocess.run's list form to avoid shell parsing issues
        cmd = ["fasttree", "-lg", "-gamma", alignment_file]
        with open(tree_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr)
        logging.info(f"Successfully generated tree for {alignment_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error generating tree for {alignment_file}: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error generating tree for {alignment_file}: {str(e)}")
        raise

def calculate_closest_distances(tree_file):
    """Calculate distances between each simulation sequence and its closest empirical sequence using branch lengths"""
    try:
        tree = read(tree_file, "newick")
        leaves = tree.get_terminals()
        empirical_leaves = [leaf for leaf in leaves if leaf.name.startswith('empirical_')]
        simulation_leaves = [leaf for leaf in leaves if leaf.name.startswith('simulation_')]
        
        if not empirical_leaves or not simulation_leaves:
            logging.warning(f"No empirical or simulation leaves found in tree: {tree_file}")
            return None
        
        # Store the distance from each simulation sequence to the closest empirical sequence
        distances = []
        
        for sim_leaf in simulation_leaves:
            min_dist = float('inf')
            
            for emp_leaf in empirical_leaves:
                try:
                    # Directly use tree.distance to calculate distance
                    distance = tree.distance(sim_leaf, emp_leaf)
                    if distance > 0:  # Only consider valid distances
                        min_dist = min(min_dist, distance)
                except Exception as e:
                    logging.warning(f"Error calculating distance between {sim_leaf.name} and {emp_leaf.name}: {str(e)}")
                    continue
            
            if min_dist != float('inf'):
                distances.append(min_dist)
            else:
                logging.warning(f"No valid distance found for simulation leaf: {sim_leaf.name}")
        
        if not distances:
            logging.warning(f"No valid distances calculated for tree: {tree_file}")
            return None
        
        return distances
    except Exception as e:
        logging.error(f"Error in calculate_closest_distances: {str(e)}")
        return None

def process_file(args):
    emp_path, sim_path, simulation_type, output_dir = args
    file_name = os.path.basename(emp_path)
    aln_dir = os.path.join(output_dir, 'combine_aln', simulation_type)
    tree_dir = os.path.join(output_dir, 'combine_tree', simulation_type)
    
    try:
        # Ensure directories exist and have correct permissions
        os.makedirs(aln_dir, exist_ok=True)
        os.makedirs(tree_dir, exist_ok=True)
        os.chmod(aln_dir, 0o755)
        os.chmod(tree_dir, 0o755)
        
        combined_aln = os.path.join(aln_dir, file_name)
        tree_file = os.path.join(tree_dir, file_name.replace('.fasta', '.tree'))
        
        # Log file paths
        logging.info(f"Processing file: {file_name}")
        logging.info(f"Combined alignment file: {combined_aln}")
        logging.info(f"Tree file: {tree_file}")
        
        combine_alignments(emp_path, sim_path, combined_aln)
        generate_tree(combined_aln, tree_file)
        
        # Verify that the tree file was generated
        if not os.path.exists(tree_file):
            raise FileNotFoundError(f"Tree file was not created: {tree_file}")
            
        distances = calculate_closest_distances(tree_file)
        
        if distances:
            avg_distance = sum(distances) / len(distances)
            return {
                'File': file_name,
                'Simulation_Type': simulation_type,
                'Average_Distance': f"{avg_distance:.4f}"
            }
        else:
            return {
                'File': file_name,
                'Simulation_Type': simulation_type,
                'Average_Distance': 'NA'
            }
    except Exception as e:
        logging.error(f"Error processing file {file_name}: {str(e)}")
        return {
            'File': file_name,
            'Simulation_Type': simulation_type,
            'Average_Distance': 'ERROR',
            'Error': str(e)
        }

def plot_results(results_data, output_dir, simulation_type):
    """Create improved distribution plot for distances"""
    try:
        # Extract valid distance data
        distances = []
        for data in results_data:
            if data['Average_Distance'] not in ['NA', 'ERROR']:
                try:
                    distances.append(float(data['Average_Distance']))
                except ValueError:
                    continue

        if not distances:
            logging.warning(f"No valid distances to plot for {simulation_type}")
            return

        distances = np.array(distances)
        # Calculate percentiles to remove outliers
        lower, upper = np.percentile(distances, [1, 99])
        filtered = distances[(distances >= lower) & (distances <= upper)]

        # Statistics
        mean = np.mean(filtered)
        median = np.median(filtered)
        std = np.std(filtered)

        # Set style
        sns.set(style="whitegrid")

        plt.figure(figsize=(10, 6))
        # Plot histogram and KDE
        sns.histplot(filtered, bins=30, kde=True, color="#4C72B0", edgecolor="black", alpha=0.7)

        # Add mean and median lines
        plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        plt.axvline(median, color='green', linestyle='-.', linewidth=2, label=f'Median: {median:.2f}')

        # Title and labels
        plt.title(f'{simulation_type} Distance Distribution', fontsize=15, fontweight='bold')
        plt.xlabel('Distance to Closest Empirical Sequence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        # Annotate sample size
        plt.text(0.98, 0.95, f'N={len(filtered)}', ha='right', va='top', transform=plt.gca().transAxes, fontsize=11)

        plt.legend()
        plt.tight_layout()

        # Save
        plot_dir = os.path.join(output_dir, 'plots', simulation_type)
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = os.path.join(plot_dir, 'distance_distribution.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Plot saved to: {plot_file}")

    except Exception as e:
        logging.error(f"Error creating plot for {simulation_type}: {str(e)}")
        plt.close()

def calculate_overall_average(results_data):
    """Calculate overall average distance from results data"""
    valid_distances = [float(data['Average_Distance']) for data in results_data 
                      if data['Average_Distance'] not in ['NA', 'ERROR']]
    if not valid_distances:
        return 'NA'
    return f"{np.mean(valid_distances):.4f}"

def plot_distance_distribution(
    distances, 
    output_path="distance_distribution.png", 
    title="Distance Distribution"
):
    """
    Plot a beautiful distance distribution histogram with KDE, mean, and median.
    Args:
        distances (list or np.ndarray): List of distance values (float).
        output_path (str): Path to save the plot image.
        title (str): Title for the plot.
    """
    distances = np.array(distances)
    # Remove outliers (1st to 99th percentile)
    lower, upper = np.percentile(distances, [1, 99])
    filtered = distances[(distances >= lower) & (distances <= upper)]
    mean = np.mean(filtered)
    median = np.median(filtered)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
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
    print(f"Plot saved to: {output_path}")

def plot_distance_distribution_cli():
    """
    Command-line interface for plotting distance distribution from a file.
    Usage: python process_alignments.py plot_dist --input distances.txt --output plot.png --title "My Title"
    """
    parser = argparse.ArgumentParser(description="Plot distance distribution from a file.")
    parser.add_argument('--input', required=True, help='Input file with one distance per line')
    parser.add_argument('--output', default='distance_distribution.png', help='Output plot file')
    parser.add_argument('--title', default='Distance Distribution', help='Plot title')
    args = parser.parse_args(sys.argv[2:])

    # Read distances from file
    distances = []
    with open(args.input) as f:
        for line in f:
            try:
                distances.append(float(line.strip()))
            except ValueError:
                continue
    if not distances:
        print("No valid distances found in input file.")
        return
    plot_distance_distribution(distances, args.output, args.title)

def plot_combined_distributions(all_distances, output_path, title="Combined Distance Distributions"):
    """
    Plot combined distance distributions for all models using FacetGrid, y-axis is frequency (count).
    """
    # Organize data as DataFrame
    data = []
    for model, dists in all_distances.items():
        for v in dists:
            data.append({'Model': model, 'Distance': v})
    df = pd.DataFrame(data)
    # Remove outliers
    df = df.groupby('Model').apply(
        lambda g: g[g['Distance'].between(*np.percentile(g['Distance'], [1, 99]))]
    ).reset_index(drop=True)
    # FacetGrid
    g = sns.FacetGrid(df, row="Model", hue="Model", aspect=4, height=1.2, sharex=True, sharey=False)
    g.map(sns.histplot, "Distance", bins=30, stat="count", element="step", fill=True, alpha=0.7)
    g.set_titles(row_template="{row_name}", size=9)
    g.set_xlabels("Distance to Closest Empirical Sequence", fontsize=12)
    g.set_ylabels("Frequency", fontsize=10)
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle(title, fontsize=16, fontweight='bold')
    for ax in g.axes.flat:
        ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)
    # Small legend
    g.add_legend(title="", adjust_subtitles=True)
    for text in g._legend.texts:
        text.set_fontsize(8)
    g._legend.set_bbox_to_anchor((1.01, 0.5))
    g._legend.set_frame_on(False)
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined plot saved to: {output_path}")

def plot_combined_boxplot(all_distances, output_path, title="Combined Distance Boxplot"):
    """
    Draw a boxplot consistent with the style of the distribution histogram, with unified color, font, and background, and do not show outlier points.
    The model order is sorted in ascending order by the number in the model name.
    """
    def extract_number(model_name):
        match = re.search(r'([\d\.]+)$', model_name)
        return float(match.group(1)) if match else 0

    # Organize data as DataFrame and remove extreme outliers
    data = []
    for model, dists in all_distances.items():
        arr = np.array(dists)
        if len(arr) > 0:
            lower, upper = np.percentile(arr, [1, 99])
            arr = arr[(arr >= lower) & (arr <= upper)]
            for v in arr:
                data.append({'Model': model, 'Distance': v})
    df = pd.DataFrame(data)

    # Sort by the number in the model name
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
    print(f"Boxplot saved to: {output_path}")

def plot_results_folder_cli():
    """
    Command-line interface for plotting all result CSVs in a results folder with subdirectories.
    Usage: python process_alignments.py plot_results_folder --input results_folder --output plots_folder
    """
    import glob
    parser = argparse.ArgumentParser(description="Plot distance distributions for all CSVs in a results folder (with subdirectories).")
    parser.add_argument('--input', required=True, help='Input folder containing result subfolders with CSV files')
    parser.add_argument('--output', required=True, help='Output folder to save plots (mirrors input structure)')
    args = parser.parse_args(sys.argv[2:])

    input_folder = args.input
    output_folder = args.output
    
    # Store all model distance data
    all_distances = {}

    # Recursively find all distance_results_*.csv files
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.startswith('distance_results_') and filename.endswith('.csv'):
                csv_file = os.path.join(dirpath, filename)
                # Read distances from CSV
                distances = []
                with open(csv_file) as f:
                    header = f.readline().strip().split(',')
                    if 'Average_Distance' not in header:
                        print(f"Skip {csv_file}: No 'Average_Distance' column.")
                        continue
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
                if not distances:
                    print(f"No valid distances in {csv_file}")
                    continue
                
                # Determine output subfolder (mirror input structure)
                rel_path = os.path.relpath(dirpath, input_folder)
                plot_dir = os.path.join(output_folder, rel_path)
                os.makedirs(plot_dir, exist_ok=True)
                
                # Save single model plot
                plot_path = os.path.join(plot_dir, 'distance_distribution.png')
                title = f"Distance distribution of {os.path.basename(dirpath)}"
                plot_distance_distribution(distances, plot_path, title)
                print(f"Plotted: {plot_path}")
                
                # Store distance data for combined plot
                model_name = os.path.basename(dirpath)
                all_distances[model_name] = distances

    # Create combined plot
    if all_distances:
        combined_plot_path = os.path.join(output_folder, 'combined_distributions.png')
        plot_combined_distributions(all_distances, combined_plot_path)
        # Additionally: generate boxplot
        boxplot_path = os.path.join(output_folder, 'combined_boxplot.png')
        plot_combined_boxplot(all_distances, boxplot_path)
    else:
        print("No valid data found to create combined plots")

def main():
    parser = argparse.ArgumentParser(description='Calculate distances between empirical and simulated sequences')
    parser.add_argument('--empirical', required=True, help='Directory containing empirical sequences')
    parser.add_argument('--simulation', required=True, nargs='+', help='One or more directories containing simulated sequences')
    parser.add_argument('--output', default='results', help='Output directory for results (default: results)')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plot')
    parser.add_argument('--threads', type=int, default=4, help='Number of parallel processes (default: 4)')
    args = parser.parse_args()

    try:
        # Ensure output directory exists
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory: {output_dir}")

        results_data = {}

        for sim_dir in args.simulation:
            simulation_type = os.path.basename(sim_dir.rstrip('/'))
            create_directories(simulation_type, output_dir)
            
            empirical_dir = os.path.abspath(args.empirical)
            simulation_dir = os.path.abspath(sim_dir)
            
            if not os.path.exists(empirical_dir):
                logging.error(f"Empirical directory does not exist: {empirical_dir}")
                continue
            if not os.path.exists(simulation_dir):
                logging.error(f"Simulation directory does not exist: {simulation_dir}")
                continue
                
            empirical_files = sorted(os.listdir(empirical_dir))
            simulation_files = sorted(os.listdir(simulation_dir))
            
            if len(empirical_files) != len(simulation_files):
                logging.warning(f"Number of files mismatch in {simulation_type}: empirical={len(empirical_files)}, simulation={len(simulation_files)}")
                continue
                
            file_args = []
            for emp_file, sim_file in zip(empirical_files, simulation_files):
                if emp_file != sim_file:
                    logging.warning(f"File name mismatch: {emp_file} != {sim_file}")
                    continue
                emp_path = os.path.join(empirical_dir, emp_file)
                sim_path = os.path.join(simulation_dir, sim_file)
                file_args.append((emp_path, sim_path, simulation_type, output_dir))
                
            results_data[simulation_type] = []
            
            # Create a progress bar with a clearer format
            pbar = tqdm(
                total=len(file_args),
                desc=f"Processing {simulation_type}",
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}',
                ncols=80,
                leave=True
            )
            
            with ProcessPoolExecutor(max_workers=args.threads) as executor:
                futures = [executor.submit(process_file, fa) for fa in file_args]
                for f in as_completed(futures):
                    result = f.result()
                    results_data[simulation_type].append(result)
                    pbar.update(1)
            
            pbar.close()

            # Save results to CSV
            results_dir = os.path.join(output_dir, 'results', simulation_type)
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(results_dir, f'distance_results_{timestamp}.csv')
            
            try:
                with open(results_file, 'w', newline='') as csvfile:
                    fieldnames = ['File', 'Simulation_Type', 'Average_Distance']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in results_data[simulation_type]:
                        writer.writerow(row)
                    
                    # Add overall average
                    valid_distances = [float(row['Average_Distance']) for row in results_data[simulation_type] 
                                    if row['Average_Distance'] not in ['NA', 'ERROR']]
                    if valid_distances:
                        overall_avg = sum(valid_distances) / len(valid_distances)
                        writer.writerow({
                            'File': 'OVERALL_AVERAGE',
                            'Simulation_Type': simulation_type,
                            'Average_Distance': f"{overall_avg:.4f}"
                        })
                logging.info(f"Results saved to: {results_file}")
            except Exception as e:
                logging.error(f"Error saving results to CSV: {str(e)}")

            # Generate plots
            if args.plot:
                try:
                    plot_results(results_data[simulation_type], output_dir, simulation_type)
                    logging.info(f"Plot generated for {simulation_type}")
                except Exception as e:
                    logging.error(f"Error generating plot: {str(e)}")

    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'plot_dist':
        plot_distance_distribution_cli()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == 'plot_results':
        plot_results_folder_cli()
        sys.exit(0)
    main() 