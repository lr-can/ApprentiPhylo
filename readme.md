# PhyloClassifier - Automated pipeline for simulation and classification
Simulation → Tree Inference → Phylogenetic Metrics → Classification → Reporting → Dashboard


# 📑 Table of Contents
1. [Introduction](#introduction)  
2. [Pipeline Overview](#pipeline-overview)  
3. [Installation & Dependencies](#installation--dependencies)  
4. [Directory Structure](#directory-structure)  
5. [Command-Line Interface](#command-line-interface)  
6. [Simulation Pipeline](#simulation-pipeline)  
   - [Step 1 — Preprocessing](#step-1--preprocessing)
   - [Step 2 — Simulation](#step-2--simulation)
   - [Step 3 — Tree Inference](#step-3--tree-inference)
   - [Step 4 — Phylogenetic Metrics](#step-4--phylogenetic-metrics)
7. [Classification Pipeline](#classification-pipeline)  
8. [Dashboard Visualization](#dashboard-visualisation)  
9. [Logging System](#logging-system)  
10. [Example Commands](#example-commands)  
11. [Troubleshooting & FAQ](#troubleshooting--faq)


# 🧬 Introduction

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

# 🧬 Pipeline Overview

=> Schéma du pipeline à faire 
    

# 🔧 Installation & Dependencies
## 1. System Requirements

This project requires:
- Python 3.8+
- Unix-based system (Linux or macOS recommended)

## 2. External tools
These external tools are **optional**.  
They are only needed if you want to use the **simulation** part of the pipeline.  
If used, they must be installed **before running the pipeline** and must be accessible in your `$PATH`.

| Tool                  | Required for                      | Description                                                    |
| --------------------- | --------------------------------- | -------------------------------------------------------------- |
| **FastTree**          | Maximum-likelihood tree inference | Builds ML phylogenetic trees from multiple sequence alignments |
| **BppSeqGen (Bio++)**   | Sequence simulation               | Required if you want to run the simulation module              |

### 2.1 FastTree 2.2
FastTree provides precompiled executables for:

- Linux 64-bit (AVX2 required)  
- Windows command-line (AVX2 required, SSE)  
- Multi-threaded executable (+OpenMP)  

You can download it from the [official FastTree website](http://www.microbesonline.org/fasttree/).

For Mac or other platforms not covered by precompiled binaries, you can compile FastTree from source [(see official instructions)](https://morgannprice.github.io/fasttree/#Install)

### 2.2 BppSeqGen
BppSeqGen is part of the [BppSuite](https://github.com/BioPP/bppsuite) (Bio++ 3.0.0)  
It can be compiled directly from the source files.

⚠️ Before compiling BppSeqGen, you must install the Bio++ libraries (e.g. in `$bpp_dir`).  
The required libraries are:

- **bpp-core**
- **bpp-seq**
- **bpp-phyl**
- **bpp-popgen**

For detailed installation instructions, see the official guide:  
https://github.com/BioPP/bpp-documentation.wiki.git

## 3. Installation
You can install the project automatically (recommended) or manually.

### 3.1 Automatic installation (recommended)

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

After installation, you can activate the environment at any time with:
```
source Project_environment/bin/activate
```

### 3.2 Manual installation

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
which FastTree
which bppseqgen
```

# 📁 Directory Structure




🖥️ Command-Line Interface

The main entrypoint:

python3 scripts/main2.py <command> [options]


Available commands:

Command	Description
simulate	Runs preprocessing → simulation → tree inference → metrics
classify	Runs the machine-learning classification pipeline
visualisation	Launches the dashboard
🧪 SIMULATION Pipeline

Triggered by:

python3 scripts/main2.py simulate [OPTIONS]


This pipeline includes 4 stages:

⭐ Step 1 — Preprocessing

Handled by:

Preprocess.preprocessing()
Preprocess.remove_gaps()
Preprocess.remove_ambig_sites()

Purpose

Clean raw alignments before simulation.

Standardize formats.

Remove sequences that do not meet criteria.

Options
Option	Explanation
--pre-input	Directory containing raw alignments.
--pre-output	Where cleaned data will be written.
--minseq	Minimum number of sequences required in each alignment.
--maxsites	Max allowed alignment length (sites).
--minsites	Min allowed alignment length.
--alphabet	"aa" or "dna"
Output

A folder:

<pre-output>/clean_data/

⭐ Step 2 — Simulation

Implemented by BppSimulator.

Purpose

Generate simulated alignments based on:

cleaned alignments,

a reference tree,

a Bio++ configuration file (.bpp),

an extinction rate parameter.

Options
Option	Description
--align	Input directory of cleaned alignments.
--tree	Reference tree file/folder used by Bio++.
--config	Bio++ simulation config file.
--sim-output	Directory to write simulated alignments.
--ext_rate	Extinction rate used in the simulation process.
Output
<sim-output>/alignment_*.fa

⭐ Step 3 — Tree Inference

Executed via:

ComputingTrees.compute_all_trees()

Purpose

Infer phylogenetic trees for each simulated alignment using the tools defined in the internal module.

Options
Option	Description
--tree-output	Output directory for inferred trees.
Output
<tree-output>/*.nwk

⭐ Step 4 — Phylogenetic Metrics (MPD)

Metrics computed for each tree using:

tree_summary()

Purpose

Compute summary statistics such as:

MPD (Mean Pairwise Distance)

number of leaves

Options
Option	Description
--metrics-output	Where the CSV containing metrics will be stored.
Output
<metrics-output>/phylo_metrics.csv

🧠 CLASSIFICATION Pipeline

Run with:

python3 scripts/main2.py classify [OPTIONS]


The pipeline includes:

Run 1 — main classification

Run 2 (optional) — refinement

Optional PDF report generation

⭐ Options
Option	Description
--real-align	Path to preprocessed real alignments.
--sim-align	Path to simulated alignments.
--output	Output directory for classification results.
--config	JSON config file for classification.
--tools	Directory containing ML models, embeddings, etc.
--two-iterations	Enables Run1 + Run2 refinement.
--threshold	Threshold used to classify simulations as REAL.
--report-output	Optional PDF report output path.
⭐ Run 1 — Main classification

Executed by:

run_classification()


Produces:

raw predictions

feature matrices

confusion matrices

probability plots

⭐ Run 2 (optional refinement)

Enabled with:

--two-iterations


Purpose:

re-train or adjust decision rules based on Run 1

improve separation real/simulated

⭐ PDF Report Generation

If you provide:

--report-output <path.pdf>


The pipeline will generate:

logistic regression training history

summary statistics

final classification report

plots

📊 Dashboard Visualisation

Run with:

python3 scripts/main2.py visualisation


This launches a Dash web application that allows interactive exploration of:

classification results

simulated datasets

metrics

confusion matrices

probabilities

📝 Logging System

Every step writes into:

logs/pipeline_log.csv


Example content:

step	status	duration	args
simulate_pipeline	success	120.5	{...}
classify_pipeline	error: missing file	0.02	{...}

Very useful for:

debugging

workflow tracking

reproducibility

🎯 Example Commands
Simulation
python3 scripts/main2.py simulate \
    --pre-input data/prot_mammals \
    --pre-output results/preprocessed \
    --minseq 5 \
    --maxsites 2000 \
    --minsites 100 \
    --alphabet aa \
    --align results/preprocessed/clean_data \
    --tree data/prot_mammals/trees \
    --config backup/config/bpp/aa/WAG_frequencies.bpp \
    --sim-output results/simulations \
    --ext_rate 0.3 \
    --tree-output results/trees \
    --metrics-output results/metrics

Classification (Run1 only)
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config backup/config_template.json \
    --tools backup/

Run1 + Run2 (refined)
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config backup/config_template.json \
    --tools backup/ \
    --two-iterations

Run1 + Run2 + PDF Report
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config backup/config_template.json \
    --tools backup/ \
    --two-iterations \
    --report-output results/classification/final_report.pdf

Dashboard
python3 scripts/main2.py visualisation

❗ Troubleshooting & FAQ
1. The simulation step fails with a Bio++ error

Check:

the .bpp config file syntax

path to bppseqgen

alphabet match (aa or dna)

2. Trees are empty or missing

Verify:

tree inference tools installed

correct file extensions for simulated alignments

3. Classification fails

Common issues:

malformed JSON config

missing ML models in --tools folder

inconsistent sequence alphabets

4. Dashboard cannot start

Make sure:

pip install dash


and that no other program occupies port 8050.

🎉 End of Documentation

If you'd like, I can also generate:

a short version,

a multi-file documentation,

a PDF version,

or diagram visualizations.

Just tell me!
