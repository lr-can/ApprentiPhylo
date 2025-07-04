# LEARNING-BASED REALISM ANALYSIS OF PHYLOGENETIC SIMULATION METHODS

**Author:** LI PENGJUN  
**Supervision:** M. GUEGUEN LAURENT, LBBE, Lyon 1  
**Contact:** PENGJUNLI2022@163.COM

This project is based on the original project: https://github.com/LIPENGJUN2022/deelogeny-m2. In my contribution, I have extended the pipeline to include additional simulations, implemented tools to calculate Mean Phylogenetic Distance (MPD), explored various classifiers, and developed a web-based bidimensional visualization to better analyze and present the results.

# Before You Start: Extract Required Data Files

Before starting the pipeline, you should extract the following archive files to set up your data and results directories:
- `data.tar.gz`
- `Original_Data.tar.gz`
- `results.tar.gz`

Use the following commands in your Linux terminal:

```bash
tar -xzf data.tar.gz
tar -xzf Original_Data.tar.gz
tar -xzf results.tar.gz
```

Make sure these files are extracted in the project root directory so that all scripts and workflows can find the necessary data.

# Simulator Installation

This project relies on the Bio++ suite of libraries and simulators. To ensure compatibility and flexibility, we recommend installing the development version of Bio++ from source, as described in the [official Bio++ installation guide](https://github.com/BioPP/bpp-documentation/wiki/Installation).

## Prerequisites

- **git** and **cmake** must be installed on your system.
- You will also need a C++ compiler (e.g., g++) and the Eigen3 library (version >= 3.8).

Install prerequisites on Ubuntu/Debian:
```bash
sudo apt update
sudo apt install git cmake g++ libeigen3-dev
```

## Step 1: Set Up Source Directory

Create a directory for the Bio++ source code:
```bash
PROJECTDIR=$HOME/devel/bpp/
mkdir -p $PROJECTDIR
cd $PROJECTDIR
```

## Step 2: Clone the Required Repositories

Clone the necessary Bio++ components:
```bash
git clone https://github.com/BioPP/bpp-core
git clone https://github.com/BioPP/bpp-seq
git clone https://github.com/BioPP/bpp-popgen
git clone https://github.com/BioPP/bpp-phyl
git clone https://github.com/BioPP/bppsuite
```

## Step 3: Compile and Install Each Component

For each component (start with `bpp-core` and `bpp-seq`), run the following commands:
```bash
cd $PROJECTDIR/bpp-core
mkdir build
cd build
cmake -B . -S .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make
make install
```
Repeat the above steps for each of the other components (`bpp-seq`, `bpp-popgen`, `bpp-phyl`, `bppsuite`).

## Step 4: Set Environment Variables

If you installed Bio++ in a non-standard location (e.g., `$HOME/.local`), add the following lines to your `~/.bashrc`:
```bash
export CPATH=$CPATH:$HOME/.local/include
export LIBRARY_PATH=$LIBRARY_PATH:$HOME/.local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
```
Then reload your shell:
```bash
source ~/.bashrc
```

## Step 5: Verify Installation

You can now use the Bio++ binaries and libraries in your project. For more details, refer to the [official Bio++ documentation](https://github.com/BioPP/bpp-documentation/wiki/Installation).

> **Note:** For every edit of the code in `/scripts` or `/src`, you should run:
> ```bash
> pip install .
> ```
> to update your environment and ensure all changes are available for import and use.

# Dataset Filtering

For this project, we require MSA (Multiple Sequence Alignment) protein family files that contain more than two species and more than two sequences per family. The information about the number of sequences and species for each family is provided in the corresponding `fam2nbseqnbspec.mono` file within each dataset folder.

To filter and select only the families that satisfy these conditions, we use the `filter_mono.py` script. This script reads the `fam2nbseqnbspec.mono` file, identifies the families meeting the criteria, and copies the corresponding MSA files to a new directory (e.g., `data/mammals_new` or `data/viridiplantae_new`).

### Usage Example

#### For mammals:
```bash
python3 scripts/filter_mono.py \
  --source-dir Original_Data/mammals/mammals_prot_results \
  --fam-file Original_Data/mammals/fam2nbseqnbspec.mono \
  --target-dir data/mammals_new
```

#### For viridiplantae:
```bash
python3 scripts/filter_mono.py \
  --source-dir Original_Data/viridiplantae/viridiplantae_prot_results \
  --fam-file Original_Data/viridiplantae/fam2nbseqnbspec.mono \
  --target-dir data/viridiplantae_new
```

- `--source-dir`: Directory containing the original MSA protein family files (e.g., `.aln` files).
- `--fam-file`: Path to the `fam2nbseqnbspec.mono` file listing the number of sequences and species for each family.
- `--target-dir`: Output directory for the filtered MSA files (e.g., `data/mammals_new` or `data/viridiplantae_new`).

Only the MSA files for families with more than two species and more than two sequences will be copied to the target directory.

# Dataset Preprocessing

After filtering the MSA protein family files, we preprocess the datasets to remove outliers, gaps, and ambiguous sites. This is done using the `scripts/preprocessing_dataset.py` script, which relies on the logic implemented in `deelogeny_m2/preprocess.py`.

### Usage Example

#### For mammals:
```bash
python3 scripts/preprocessing_dataset.py \
  --input data/mammals_new \
  --output results/preprocess_mammals_new \
  --minseq 5 \
  --maxsites 10000 \
  --minsites 10 \
  --type aa
```

#### For viridiplantae:
```bash
python3 scripts/preprocessing_dataset.py \
  --input data/viridiplantae_new \
  --output results/preprocess_viridiplantae_new \
  --minseq 5 \
  --maxsites 10000 \
  --minsites 10 \
  --type aa
```

- `--input`: Directory containing the filtered MSA files (e.g., `data/mammals_new`).
- `--output`: Directory where the preprocessing results will be saved (e.g., `results/preprocess_mammals_new`).
- `--minseq`: Minimum number of sequences required to keep an alignment.
- `--maxsites`: Maximum number of sites (columns) allowed in an alignment.
- `--minsites`: Minimum number of sites (columns) required in an alignment.
- `--type`: Sequence type, either `aa` (amino acid) or `dna`.

### Output Directory Structure

After running the preprocessing script, the output directory (e.g., `results/preprocess_mammals_new`) will contain several subfolders:

- `clean_data/`: Alignments that passed the basic filtering (length and sequence count).
- `gap_less/`: Alignments with all columns containing gaps removed.
- `ambigless/`: Alignments with ambiguous sites removed (from clean_data).
- `gap_and_ambigless/`: Alignments with both gaps and ambiguous sites removed (recommended for downstream analysis).
- `preprocess.log`: Log file summarizing the filtering and cleaning steps.

**Note:** For downstream analyses, we recommend using the files in the `gap_and_ambigless/` subdirectory, as these have had both gaps and ambiguous sites removed, ensuring the highest data quality.

# Infer Phylogenetic Trees

After preprocessing, you can infer phylogenetic trees for each protein family alignment using the `scripts/compute_tree.py` script, which utilizes the logic in `deelogeny_m2/computing_trees.py`. The underlying tree inference is performed using the FastTree program.

### Usage Example

#### For mammals:
```bash
python3 scripts/compute_tree.py \
  --input results/preprocess_mammals_new/gap_and_ambigless \
  --output results/trees_mammals_new \
  --alphabet aa
```

#### For viridiplantae:
```bash
python3 scripts/compute_tree.py \
  --input results/preprocess_viridiplantae_new/gap_and_ambigless \
  --output results/trees_viridiplantae_new \
  --alphabet aa
```

- `--input`: Directory containing the preprocessed alignments (recommended: `gap_and_ambigless` subfolder).
- `--output`: Directory where the inferred phylogenetic trees will be saved.
- `--alphabet`: Sequence type, `aa` for amino acids or `nt` for nucleotides.
- `--only`: (Optional) Path to a file listing specific alignment filenames to process (one per line).

### Underlying Command

The script uses FastTree to infer trees. For amino acid alignments, the command is:
```bash
fasttree -lg -gamma <alignment_file>
```
For nucleotide alignments, the command is:
```bash
fasttree -gtr -gamma -nt <alignment_file>
```

Each output tree is saved in Newick format with the suffix `_tree.nwk` in the specified output directory.

You can also use the provided `computing_trees.sh` script to batch process both mammals and viridiplantae datasets automatically.

# Simulation (Bppsimulator)

You can simulate sequence alignments using the Bppsimulator, which is accessible via the `scripts/simulation.py` script. This tool allows you to generate simulated alignments based on phylogenetic trees and various evolutionary models using the Bio++ suite.

### Usage Example

```bash
python3 scripts/simulation.py \
  --simulator BPP \
  --align <reference_alignment_dir> \
  --tree <tree_dir> \
  --output <output_dir> \
  --tools <path_to_tools> \
  --config <config_file1> <config_file2> ... \
  --external_branch_length <length> \
  --root_length <length> \
  --modelmapping <model_mapping_dir> \
  --gap True
```

#### Key Arguments
- `--simulator`: Must include `BPP` to use the Bppsimulator.
- `--align`: Directory containing reference alignments (e.g., from `gap_and_ambigless`).
- `--tree`: Directory containing phylogenetic trees (e.g., from `results/trees_mammals_new`).
- `--output`: Output directory for simulated alignments.
- `--tools`: Path to necessary tools (e.g., Bio++ binaries or Apptainer/Singularity images).
- `--config`: One or more BPP configuration files specifying simulation parameters.
- `--external_branch_length`: (Optional) Length to set for external branches in the tree.
- `--root_length`: (Optional) Length to set for the root branch in the tree.
- `--modelmapping`: (Optional) Directory containing model mapping files.
- `--gap`: Whether to add gaps to simulated alignments (`True` or `False`).

#### Example
```bash
python3 scripts/simulation.py \
  --simulator BPP \
  --align results/preprocess_mammals_new/gap_and_ambigless \
  --tree results/trees_mammals_new \
  --output results/mammals_simulations \
  --tools $HOME/.local/bin \
  --config config/bpp/aa/WAG_frequencies.bpp \
  --external_branch_length 0.1 \
  --root_length 0.05 \
  --gap True
```

This will generate simulated alignments using the Bppsimulator, with the specified evolutionary model and tree modifications. The output will be saved in the specified directory for downstream analysis.

## Step 0: Write the Configuration File for Simulation

Before running the Bppsimulator, you need to prepare a configuration file that specifies the evolutionary model, input alignment, tree, and simulation parameters. This file is required by the simulator to know how to generate the simulated alignments.

Below is an example configuration file for the DSO78 model (`config/bpp/aa/classic/DSO78_frequencies.bpp`):

```ini
alphabet = Protein

# Input data: alignment file to use as the root sequence
input.data1=alignment(file=align_path, format=Fasta(extended=yes, strict_name=no), sites_to_use = all, max_gap_allowed = 50%, max_unresolved_allowed = 100%)

# Evolutionary tree
input.tree1=user(file=tree_path,format=Newick)

root_freq1 = Fixed(init=observed, data=1)

# Model of evolution: Jukes-Cantor 1969 (JC69)
model1 = DSO78+F(frequencies=Fixed(init=observed, data=1))

#rate distribution
rate_distribution1 = Constant()

# Root frequencies: use the stationary frequencies of the model
process1 = Homogeneous(model=1, rate=1, tree=1, root_freq=1)

#simul 
simul1 = simul(process=1, output.sequence.file = output_path, output.sequence.format = Fasta, output.internal.sequences = false, number_of_sites = nseq)
```

- `alphabet = Protein`: Specifies the type of sequences (protein).
- `input.data1`: Defines the input alignment file and its format.
- `input.tree1`: Specifies the evolutionary tree in Newick format.
- `model1`: Sets the evolutionary model (here, DSO78 with observed frequencies).
- `simul1`: Defines the simulation process and output format.

You can modify this template for other models or simulation scenarios as needed. Save your configuration files in the appropriate directory (e.g., `config/bpp/aa/classic/`).

# Simulation Examples and Model Naming Conventions

In this project, several evolutionary models and simulation strategies are used. The main models include:
- **DSO78**
- **JTT92**
- **LG08**
- **WAG**

These models are used to simulate sequence evolution under different assumptions. The simulation scripts and output directories use a set of suffixes to indicate the specific simulation strategy or parameterization. Here is a guide to the naming conventions:

### Model Suffixes and Their Meanings
- **_frequencies** or **_F**: Use the observed amino acid frequencies from the MSA (multiple sequence alignment) to generate the root sequence, replacing the default frequencies in the model.
- **sampling_seq** or **_EP**: Sample the root sequence by position (i.e., sample each site independently according to the observed frequencies at that position in the MSA).
- **_data2**: Randomly select another MSA from the same dataset to generate the root sequence, instead of using the current family.
- **_posterior** or **_P**: Use a posterior method to simulate the sequences, typically involving more sophisticated statistical inference.
- **extra_length** or **_E**: Add extra length to the branches of the tree during simulation.
- **_root** or **_R**: Add a specific length to the root branch of the tree during simulation.

### Example Naming Interpretations
- `WAG_frequencies`: Simulations using the WAG model, with root sequence generated using observed MSA frequencies.
- `LG08_frequencies_sampling_seq`: Simulations using the LG08 model, with root sequence sampled by position.
- `JTT92_frequencies_data2`: Simulations using the JTT92 model, with root sequence generated from a randomly selected MSA in the dataset.
- `WAG_frequencies_posterior_extra_length`: Simulations using the WAG model, posterior method, and extra branch length added to the tree.
- `WAG_sampling_seq_root_0.1`: Simulations using the WAG model, root sequence sampled by position, and root branch length set to 0.1.

These conventions are reflected in the simulation shell scripts (e.g., in the `mammals/` directory) and in the output directory names. They help you quickly identify the simulation strategy and parameters used for each set of results.

# Batch Running All Simulation Scripts

To efficiently run all simulation scenarios for both mammals and viridiplantae, you can use the provided `run_all_simulations.sh` script. This script will automatically open a new terminal window for each simulation shell script found in the `mammals/` and `viridiplantae/` directories, and execute them sequentially with a short delay between each launch.

### Usage

```bash
bash run_all_simulations.sh
```

This will:
- Loop through all `.sh` scripts in the `mammals/` directory and run each in a new terminal.
- Then loop through all `.sh` scripts in the `viridiplantae/` directory and run each in a new terminal.

Each simulation script corresponds to a different model or simulation scenario (see the section on model naming conventions for details). This approach allows you to launch all simulations in parallel, making full use of your system's resources.

**Note:**
- Make sure you have the necessary permissions to execute the scripts and that your system supports opening multiple terminal windows (the script uses `gnome-terminal`).
- You can monitor the progress and output of each simulation in its own terminal window.

# Mean Phylogenetic Distance (MPD) Analysis

The Mean Phylogenetic Distance (MPD) is a metric used to quantify the average evolutionary distance between simulated and empirical sequences in a phylogenetic tree. In this project, MPD is calculated by combining empirical and simulated alignments, inferring a tree, and then measuring the distance from each simulated sequence to its closest empirical sequence.

This analysis is performed using the `scripts/MPD/process_alignments.py` script, which automates the process of combining alignments, building trees, and calculating distances.

## How MPD is Calculated
1. For each protein family, the empirical and simulated alignments are combined into a single alignment file.
2. A phylogenetic tree is inferred from the combined alignment using FastTree.
3. For each simulated sequence, the distance to the closest empirical sequence is computed using the branch lengths in the tree.
4. The mean of these distances is reported as the MPD for that family/model.

## Usage Example

### For mammals (single model):
```bash
python scripts/MPD/process_alignments.py \
  --empirical results/preprocess_mammals_new/gap_and_ambigless \
  --simulation results/mammals_simulations/BPP/WAG_frequencies \
  --output results/MPD/mammals_results/WAG_frequencies \
  --plot --threads 5
```

### For viridiplantae (single model):
```bash
python scripts/MPD/process_alignments.py \
  --empirical results/preprocess_viridiplantae_new/gap_and_ambigless \
  --simulation results/viridiplantae_simulations/BPP/WAG_frequencies \
  --output results/MPD/viridiplantae_results/WAG_frequencies \
  --plot --threads 5
```

- `--empirical`: Directory containing empirical alignments (recommended: `gap_and_ambigless` subfolder).
- `--simulation`: Directory containing simulated alignments for a specific model.
- `--output`: Output directory for MPD results and plots.
- `--plot`: (Optional) Generate distribution plots for the calculated distances.
- `--threads`: (Optional) Number of parallel processes to use.

### Batch Analysis
You can also use the provided shell scripts (e.g., `calcule_distance_in_tree/runs_mammals.sh`, `calcule_distance_in_tree/runs_viridiplantae.sh`) to batch process all models and scenarios. These scripts will output all results to the `results/MPD/` directory.

### Output
- Results are saved as CSV files in the specified output directory, with the mean and distribution of distances for each family/model.
- Plots of the distance distributions are also generated if `--plot` is specified.

This MPD analysis allows you to quantitatively compare the evolutionary similarity between simulated and empirical data across different models and simulation strategies.

# MPD Batch Analysis and Plotting Scripts

To facilitate large-scale MPD (Mean Phylogenetic Distance) analysis and visualization, several batch scripts and utilities are provided. These scripts automate the process of running MPD calculations and generating summary plots for all models and simulation scenarios.

## Overview of Scripts

- **runs_mammals.sh**: Batch processes all relevant simulation models for mammals, calculates MPD, and generates result CSVs and plots in `results/MPD/mammals_results/`.
- **runs_viridiplantae.sh**: Batch processes all relevant simulation models for viridiplantae, calculates MPD, and generates result CSVs and plots in `results/MPD/viridiplantae_results/`.
- **runs_3_model.sh**: Quickly runs MPD analysis for three basic models (DSO78, JTT92, LG08) for viridiplantae, outputting to `results/MPD/viridiplantae_results/`.
- **plot_mammals.sh**: Batch generates summary plots for all result folders in `results/MPD/mammals_results/`.
- **plot_viridiplantae.sh**: Batch generates summary plots for all result folders in `results/MPD/viridiplantae_results/`.
- **create_folders.py**: Organizes and groups result folders for viridiplantae into logical groups in `results/MPD/viridiplantae_group_results/` for comparative analysis.
- **plot_groups.py**: Generates combined distribution and boxplot visualizations for each group in `results/MPD/viridiplantae_group_results/`, saving plots to `results/MPD/viridiplantae_group_plots/`.

## Example Usage

### Batch MPD Calculation for Mammals
```bash
bash calcule_distance_in_tree/runs_mammals.sh
```
This will process all major simulation scenarios for mammals and save results and plots in `results/MPD/mammals_results/`.

### Batch MPD Calculation for Viridiplantae
```bash
bash calcule_distance_in_tree/runs_viridiplantae.sh
```
This will process all major simulation scenarios for viridiplantae and save results and plots in `results/MPD/viridiplantae_results/`.

### Quick MPD Calculation for Three Models (Viridiplantae)
```bash
bash calcule_distance_in_tree/runs_3_model.sh
```
This will process DSO78, JTT92, and LG08 models for viridiplantae.

### Batch Plotting for Mammals
```bash
bash calcule_distance_in_tree/plot_mammals.sh
```
This will generate summary plots for all mammals MPD results.

### Batch Plotting for Viridiplantae
```bash
bash calcule_distance_in_tree/plot_viridiplantae.sh
```
This will generate summary plots for all viridiplantae MPD results.

### Organize and Group Viridiplantae Results
```bash
python scripts/MPD/create_folders.py
```
This will create grouped result folders in `results/MPD/viridiplantae_group_results/` for comparative analysis.

### Plot Grouped Results for Viridiplantae
```bash
python scripts/MPD/plot_groups.py
```
This will generate combined distribution and boxplots for each group, saving them in `results/MPD/viridiplantae_group_plots/`.

## Notes
- All scripts assume the standard project directory structure as described above.
- Ensure all dependencies (Python packages, FastTree, etc.) are installed and available in your environment.
- For large datasets, adjust the number of threads in the shell scripts for optimal performance.

# Classifiers: 

The classifiers and related scripts are located in the `simulations-classifiers` directory. The recommended way to install and set up this part of the project is to use [uv](https://github.com/astral-sh/uv), a fast Python package and environment manager.

## Installation Steps

1. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/)**
   - Follow the official instructions to install `uv` on your system.

2. **Clone this repository**
   - If you have not already done so, clone the repository to your local machine.

3. **Navigate to the `simulations-classifiers` directory**
   ```bash
   cd simulations-classifiers
   ```

4. **Install dependencies using `uv`**
   - Run the following command at the root of the `simulations-classifiers` directory:
   ```bash
   uv sync
   ```
   - This will automatically install the required Python version and all necessary packages as specified in the project configuration files.

5. **Run Python or Jupyter Lab with the project environment**
   - To launch a Python interpreter with all dependencies:
   ```bash
   uv run python
   ```
   - To launch a Jupyter Lab notebook server:
   ```bash
   uv run jupyter lab
   ```

After completing these steps, you will have a fully functional environment for running the classifiers and related scripts in the `simulations-classifiers` directory.

## Running Classifiers on a Cluster

To run the classifiers efficiently on a computing cluster, follow these steps:

1. **Prepare Configuration Files**
   - Place all your configuration files (e.g., `.json` files) in the `config` directory under `simulations-classifiers`.

2. **Generate Command and SLURM Job Files**
   - Run the following script to automatically generate `commands.txt` (which contains all the commands to run the classifiers) and a SLURM job array script (`slurm_job_array.slurm`):
   ```bash
   bash generate_command.sh
   ```
   - This script will scan the `config` directory for all configuration files, create a command for each, and write them to `commands.txt`. It will also generate a SLURM job array script suitable for batch submission.

3. **Submit the SLURM Job Array**
   - Submit the generated SLURM script to your cluster's job scheduler:
   ```bash
   sbatch slurm_job_array.slurm
   ```
   - This will launch a job array, running each classifier configuration as a separate task on the cluster, utilizing available GPUs and resources as specified in the script.

**Note:**
- Make sure the `logs/` directory exists for output and error logs, or create it with `mkdir -p logs`.
- Adjust resource parameters in `slurm_job_array.slurm` as needed for your cluster environment.

## Renaming Result Folders for Clearer Plots

To make the results easier to interpret and visualize in plots, you can use the provided folder renaming scripts to standardize and clarify the names of result directories. This is especially useful when you have many simulation/model result folders with complex or inconsistent names.

### Python Script: `rename_folders.py`
- This script recursively traverses specified root directories and renames subfolders according to a set of rules, producing standardized, concise names (e.g., `WAG_F_P_E_0.1`, `WAG_EP_DATA2_R_0.05`).
- The renaming rules are designed to reflect the model and simulation parameters in a consistent format, making it easier to group and compare results in downstream analyses and plots.
- To use:
  ```bash
  python rename_folders.py
  ```
- The script will process all roots defined in the script (such as `runs_viridiplantae`, `runs_mammals`, etc.).

### Bash Script: `rename_folders.sh`
- This shell script provides a template for renaming folders in `runs_viridiplantae` according to common patterns, especially for WAG model variants.
- It uses pattern matching and string manipulation to rename folders to the standardized format used in plotting scripts.
- To use:
  ```bash
  bash rename_folders.sh
  ```
- You can modify or extend this script to match your specific folder naming patterns.

**Why Rename?**
- Standardized folder names make it much easier to select, group, and display results in summary plots.
- The plotting scripts expect or benefit from these concise, consistent names for grouping and labeling.

## Visualization Scripts

After running and (optionally) renaming your result folders, you can use the provided visualization scripts in the `/simulations-classifiers` directory to generate summary plots and figures for your classification results. **All output figures will be saved in `results/simulations-classifiers/visualization/` and its subdirectories.**

### 1. Confusion Matrix Visualization
- **Script:** `simulations-classifiers/visualize_confusion_matrix.py`
- **Purpose:** Generates confusion matrix plots for different simulation groups and classifiers.
- **Usage:**
  ```bash
  python simulations-classifiers/visualize_confusion_matrix.py --runs_path runs_viridiplantae
  ```
- The output will be saved in `results/simulations-classifiers/visualization/confusion_matrix/`.

### 2. Heatmap Visualization
- **Script:** `simulations-classifiers/visualize_heatmap.py`
- **Purpose:** Creates heatmaps of accuracy and F1 score for WAG model variants across classifiers.
- **Usage:**
  ```bash
  python simulations-classifiers/visualize_heatmap.py
  ```
- The output will be saved in `results/simulations-classifiers/visualization/wag_heatmaps/`.

### 3. Training/Validation Curve Visualization
- **Script:** `simulations-classifiers/visualize_train_valid.py`
- **Purpose:** Plots training and validation loss, accuracy, and F1 score curves for all classifiers and simulation types.
- **Usage:**
  ```bash
  python simulations-classifiers/visualize_train_valid.py
  ```
- The output will be saved in `results/simulations-classifiers/visualization/train_valid_plot/`.

**Note:**
- Make sure your result folders (e.g., `runs_viridiplantae`) are named according to the conventions described above for best results in the plots.
- You can modify the scripts to point to other result directories as needed.
- All visualization outputs are collected under `results/simulations-classifiers/visualization/` for easy access and organization.

# Bidimensional Visualization Web App

This project provides an interactive web application for visualizing the relationship between phylogenetic distance and classifier accuracy across different simulation scenarios and groups. This visualization helps you compare the performance of different classifiers and simulation settings in a two-dimensional (2D) space.

## Folder Name Standardization

Before using the visualization, it is recommended to standardize the names of your result folders for clarity and consistency. Use the provided `plots_two_dimension/rename_folders.py` script:

```bash
python plots_two_dimension/rename_folders.py
```
This script will recursively rename folders in your results directory (e.g., `results/MPD/viridiplantae_group_results`) to a concise and consistent format, making it easier to group and display results in the visualization.

## Launching the Visualization

The main visualization app is implemented in `plots_two_dimension/app.py` using Dash and Plotly. It reads the processed and renamed results and allows you to interactively explore the data in your web browser.

### Data Paths Used
- **Group results:** `results/MPD/viridiplantae_group_results` (after renaming)
- **Classifier results:** `results/simulations-classifiers/runs_viridiplantae`

### How to Run
1. Make sure you have installed the required Python packages (see `plots_two_dimension/requirements.txt`).
2. Standardize your folder names using `rename_folders.py` as described above.
3. Launch the web app:
   ```bash
   python plots_two_dimension/app.py
   ```
4. Open your browser and go to `http://127.0.0.1:8050/` to interact with the visualization.

### Features
- Select groups, classifiers, and subgroups (with/without data2) to display.
- Each point represents a simulation scenario and classifier, with:
  - **Marker shape** for group
  - **Marker color** for classifier
  - **Marker size** for simulation suffix (e.g., E_0, E_0.05, R_0, etc.)
- Optionally connect points by lines in order of simulation suffix.
- Hover to see detailed labels for each point.

This tool provides a powerful way to visually compare the effects of different simulation parameters and classifier choices on phylogenetic distance and classification accuracy.

# Pipeline Summary: Step-by-Step Workflow

Below is a concise summary of the main steps to run the full pipeline, from raw data to bidimensional visualization. Each step lists the key script or command to use.

1. **Prepare Raw Data**
   - Place your original MSA protein family files and metadata (e.g., fam2nbseqnbspec.mono) in the appropriate directory (e.g., `Original_Data/mammals/` or `Original_Data/viridiplantae/`).

2. **Filter Families**
   - Select families with more than two species and more than two sequences:
     ```bash
     python3 scripts/filter_mono.py --source-dir <input_dir> --fam-file <fam2nbseqnbspec.mono> --target-dir <filtered_dir>
     ```

3. **Preprocess Alignments**
   - Remove outliers, gaps, and ambiguous sites:
     ```bash
     python3 scripts/preprocessing_dataset.py --input <filtered_dir> --output <preprocess_dir> --minseq 5 --maxsites 10000 --minsites 10 --type aa
     ```

4. **Infer Phylogenetic Trees**
   - Build trees for each alignment:
     ```bash
     python3 scripts/compute_tree.py --input <preprocess_dir>/gap_and_ambigless --output <tree_dir> --alphabet aa
     ```

5. **Write Simulation Configuration File**
   - Prepare a config file for each simulation scenario (see `config/bpp/aa/classic/DSO78_frequencies.bpp` for an example).
   - If you want to run simulations on a cluster, you can generate all simulation commands and a SLURM job array script automatically:
     ```bash
     cd simulations-classifiers
     bash generate_command.sh
     sbatch slurm_job_array.slurm
     ```
   - This will scan your config directory, create all necessary commands, and submit them as a job array to your cluster's scheduler. Adjust resource parameters in the SLURM script as needed for your environment.

6. **Run Simulations**
   - Simulate alignments using the Bppsimulator:
     ```bash
     python3 scripts/simulation.py --simulator BPP --align <preprocess_dir>/gap_and_ambigless --tree <tree_dir> --output <sim_output_dir> --tools <path_to_tools> --config <config_file>
     ```

7. **Run Classifiers**
   - Train and evaluate classifiers on simulated data (see `simulations-classifiers/README.md` for details).

8. **Calculate MPD (Mean Phylogenetic Distance)**
   - Compare simulated and empirical data:
     ```bash
     python3 scripts/MPD/process_alignments.py --empirical <preprocess_dir>/gap_and_ambigless --simulation <sim_output_dir> --output <mpd_output_dir> --plot
     ```

9. **Organize and Rename Result Folders**
   - Standardize result folder names for visualization:
     ```bash
     python plots_two_dimension/rename_folders.py
     ```

10. **Launch the Bidimensional Visualization Web App**
    - Explore results interactively in your browser:
      ```bash
      python plots_two_dimension/app.py
      # Then open http://127.0.0.1:8050/ in your browser
      ```

This pipeline allows you to go from raw data to interactive, publication-ready visualizations step by step. Adjust parameters and paths as needed for your specific datasets and analysis goals.
