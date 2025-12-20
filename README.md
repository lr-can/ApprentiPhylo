# PhyloClassifier - Automated pipeline for simulation and classification
Simulation → Tree Inference → Phylogenetic Metrics → Classification → Reporting → Dashboard


# Table of Contents
1. [Introduction](#introduction)  
2. [Pipeline Overview](#pipeline-overview)  
3. [Installation & Dependencies](#installation--dependencies)  
4. [Directory Structure](#directory-structure)  
5. [Command-Line Interface](#command-line-interface)  
6. [Simulation Pipeline](#simulation-pipeline)  
7. [Classification Pipeline](#classification-pipeline)  
8. [Dashboard Visualization](#dashboard-visualisation)  
9. [Logging System](#logging-system)  
10. [Example Commands](#example-commands)  
11. [Troubleshooting & FAQ](#troubleshooting--faq)


# 1. Introduction

This repository provides a unified end-to-end phylogenetic pipeline designed to:
- preprocess real biological alignments,
- simulate new datasets based on those alignments and evolutionary models,
- infer phylogenetic trees from real and simulated data,
- compute phylogenetic metrics (such as MPD),
- classify simulated vs. real alignments using ML tools,
- generate PDF reports summarizing classification results,
- visualize results through an optional dashboard.

The entire workflow is controlled through a single Python entrypoint:
`python3 scripts/main2.py <command> [options]`

# 2. Pipeline Overview

=> Schéma du pipeline à faire 
    

# 3. Installation & Dependencies
## 3.1. System Requirements

This project requires:
- Python 3.8+
- Unix-based system (Linux or macOS recommended)

## 3.2. External tools
These external tools are **optional**.  
They are only needed if you want to use the **simulation** part of the pipeline.  
If used, they must be installed **before running the pipeline** and must be accessible in your `$PATH`.

| Tool                  | Required for                      | Description                                                    |
| --------------------- | --------------------------------- | -------------------------------------------------------------- |
| **FastTree**          | Maximum-likelihood tree inference | Builds ML phylogenetic trees from multiple sequence alignments |
| **BppSeqGen (Bio++)**   | Sequence simulation               | Required if you want to run the simulation module              |

### 3.2.1 FastTree 2.2
FastTree provides precompiled executables for:

- Linux 64-bit (AVX2 required)  
- Windows command-line (AVX2 required, SSE)  
- Multi-threaded executable (+OpenMP)  

You can download it from the [official FastTree website](http://www.microbesonline.org/fasttree/).

For Mac or other platforms not covered by precompiled binaries, you can compile FastTree from source [(see official instructions)](https://morgannprice.github.io/fasttree/#Install)

### 3.2.2 BppSeqGen
BppSeqGen is part of the [BppSuite](https://github.com/BioPP/bppsuite) (Bio++ 3.0.0)  
It can be compiled directly from the source files.

/!\ Before compiling BppSeqGen, you must install the Bio++ libraries (e.g. in `$bpp_dir`).  
The required libraries are:

- **bpp-core**
- **bpp-seq**
- **bpp-phyl**
- **bpp-popgen**

For detailed installation instructions, see the official guide:  
https://github.com/BioPP/bpp-documentation/wiki/Installation

## 3.3. Installation
You can install the project automatically (recommended) or manually.

### 3.3.1 Automatic installation (recommended)

Clone the repository:
```
git clone <URL_of_this_repository>
cd <repository_name>
```

Then run the installation script:
```
bash install.sh
```

This script will:
- Create a dedicated Python virtual environment in Project_environment/
- Activate the environment
- Install all required Python dependencies from requirements.txt

/!\ After installation, the environment is automatically deactivated. 
You can activate the it at any time with:
```
source Project_environment/bin/activate
```

### 3.3.2 Manual installation

If you prefer to install everything manually:

- Clone the repository
```
git clone <URL_of_this_repository>
cd <repository_name>
```

- Create and activate a virtual environment
```
python3 -m venv .venv
source Project_environment/bin/activate
```

- Install Python dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

- (Optional) Ensure external tools are accessible:
```
which FastTree    mv FastTree fasttree
which bppseqgen
```

# Directory Structure
The following directory structure illustrates the recommended organization of the project:

```text
ApprentiPhylo/
├── config/
│   ├── bpp/
│   ├── yaml/
│   └── config_template.json
│
├── data/              # Optional directory for storing input data
│
├── scripts/
│   ├── main.py
│   ├── preprocess.py
│   ├── simulation.py
│   ├── compute_tree.py
│   ├── classification.py
│   ├── analyse_classif.py
│   ├── analyse_predictions.py
│   ├── analyse_predictions_plotly.py
│   ├── phylo_metrics.py
│   ├── filter_mono.py
│   ├── visualize_optimal_threshold.py
│   ├── dashboard.py
│   ├── report.py
│   ├── data_description.py
│   ├── data_description2.py
│   └── fix_logreg_history.py
│
├── results/
│   ├── preprocessed/
│   ├── simulations/
│   ├── classification/
│   └── trees/
│
└── README.md
```


# 4. Usage — Command-Line Interface
The project provides a unified command-line interface to run the different stages of the phylogenetic pipeline:
**simulation**, **metrics computation**, **classification**, and **visualisation**.

All commands are executed through the main entrypoint:

```bash
python3 scripts/main.py <command> [options]
```
The available commands are:

- `simulate` – Run the simulation pipeline
- `metrics` – Compute phylogenetic metrics between real and simulated data
- `classify` – Run the classification pipeline
- `visualisation` – Launch the interactive dashboard

## 4.1 General Configuration

For the simulate and classify commands, two configuration modes are supported:

### 4.1.1. YAML-based configuration (recommended)

All parameters are defined in a YAML file.
When using `--yaml`, no other command-line options are allowed.

### 4.2.2. Command-line arguments

All required options must be explicitly provided via flags.

/!\ Mixing `--yaml` with other flags is not allowed and will result in an error.


## 4.2 Simulation Pipeline

The simulate command runs the complete simulation workflow:

1. Preprocessing of input alignments

2. Sequence simulation using Bio++ (BPP)

3. Phylogenetic tree inference

▪️ Command
```
python3 scripts/main.py simulate [options]
```

▪️ Option 1 — Using a YAML configuration
```
python3 scripts/main.py simulate --yaml config/simulate.yaml
```

The YAML file must contain a simulate section defining all required parameters.

▪️ Option 2 — Using command-line arguments
```
python3 scripts/main.py simulate \
  --pre-input data/prot_mammals \
  --pre-output results/preprocessed \
  --minseq 5 \
  --maxsites 2000 \
  --minsites 100 \
  --alphabet aa \
  --align results/preprocessed/clean_data \
  --tree data/prot_mammals/trees \
  --config config/bpp/aa/WAG_frequencies.bpp \
  --sim-output results/simulations \
  --ext_rate 0.3 \
  --tree-output results/trees
```

▪️ Simulation Options

| Option          | Description                                  |
| --------------- | -------------------------------------------- |
| `--yaml`        | Path to YAML configuration file              |
| `--pre-input`   | Input directory containing raw alignments    |
| `--pre-output`  | Output directory for preprocessed alignments |
| `--minseq`      | Minimum number of sequences required         |
| `--maxsites`    | Maximum number of alignment sites            |
| `--minsites`    | Minimum number of alignment sites            |
| `--alphabet`    | Sequence type: `aa` or `dna`                 |
| `--align`       | Directory containing cleaned alignments      |
| `--tree`        | Directory containing reference trees         |
| `--config`      | BPP configuration file                       |
| `--sim-output`  | Output directory for simulated alignments    |
| `--ext_rate`    | Extinction rate parameter for simulation     |
| `--tree-output` | Output directory for inferred trees          |


## 4.3 Metrics Computation

The metrics command computes phylogenetic metrics (e.g. MPD) between empirical and simulated alignments.

▪️ Command
```
python3 scripts/main.py metrics [options]
```

Example
```
python3 scripts/main.py metrics \
  --empirical results/preprocessed/clean_data \
  --simulation results/simulations \
  --output results \
  --threads 8
```

▪️ Metrics Options
| Option         | Description                                |
| -------------- | ------------------------------------------ |
| `--empirical`  | Directory containing empirical FASTA files |
| `--simulation` | Directory containing simulated FASTA files |
| `--output`     | Output directory (default: `results`)      |
| `--threads`    | Number of parallel processes (default: 4)  |

The results are written to:

'results/metrics_results/mpd_results.csv'

## 4.4. Classification Pipeline

The classify command runs the classification workflow to distinguish real vs simulated alignments.
It supports:

- One-pass classification (Run 1)
- Two-pass refinement (Run 1 + Run 2)
- Optional PDF report generation

▪️ Command
```
python3 scripts/main2.py classify [options]
```

▪️ Option 1 — Using a YAML configuration
```
python3 scripts/main.py classify --yaml config/classify.yaml
```

▪️ Option 2 — Using command-line arguments
```
Run 1 only
python3 scripts/main.py classify \
  --real-align results/preprocessed/clean_data \
  --sim-align results/simulations \
  --output results/classification \
  --config config/config_template.json \
  --tools tools/
```
```
Run 1 + Run 2 (refinement)
python3 scripts/main.py classify \
  --real-align results/preprocessed/clean_data \
  --sim-align results/simulations \
  --output results/classification \
  --config config/config_template.json \
  --tools tools/ \
  --two-iterations
```
```
Run 1 + Run 2 + PDF report
python3 scripts/main.py classify \
  --real-align results/preprocessed/clean_data \
  --sim-align results/simulations \
  --output results/classification \
  --config config/config_template.json \
  --tools tools/ \
  --two-iterations \
  --report-output results/classification/final_report.pdf
```

▪️ Classification Options
| Option             | Description                                      |
| ------------------ | ------------------------------------------------ |
| `--yaml`           | Path to YAML configuration file                  |
| `--real-align`     | Directory containing real (empirical) alignments |
| `--sim-align`      | Directory containing simulated alignments        |
| `--output`         | Output directory for classification results      |
| `--config`         | Classification configuration file                |
| `--tools`          | Directory containing external tools              |
| `--two-iterations` | Enable Run 1 + Run 2 refinement                  |
| `--threshold`      | Classification threshold (default: 0.5)          |
| `--report-output`  | Optional output path for a PDF report            |

During the classification stage:
- Interactive Plotly visualizations are generated automatically
- Logistic regression training history is computed if a report is requested


## 4.5 Visualisation Dashboard

The visualisation command launches an interactive Dash dashboard to explore classification results.

▪️ Command
```
python3 scripts/main.py visualisation
```

This starts a local web server displaying interactive plots and summaries.

## 4.6 Logging

All major pipeline steps (simulate, classify) are logged in:

`logs/pipeline_log.csv`


Each entry includes:
- Step name
- Execution status
- Runtime duration
- Command-line arguments used
