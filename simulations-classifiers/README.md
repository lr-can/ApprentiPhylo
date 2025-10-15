# Simulations project classifiers

This repository contains a Python package implementing several classifiers for the simulations project.

## TODO

-   [ ] Add masking for padding in attention model
-   [ ] Remove first or first two amino acids from empirical proteins ?

## Installation

The recommended way to install this project is to use [uv](https://github.com/astral-sh/uv):

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone this repository
3. Run `uv sync` at the root of the cloned repository to install the required python version and packages

After that, you can run `uv run python` inside the project folder to launch a python interpreter with all needed dependencies available. You can also run `uv run jupyter lab` to launch a corresponding Jupyter Lab notebook server.

## Usage

This repository provides a `classifiers` Python package which can be used in three different ways:

1. Each classifier can be used independently from a Python script or notebook
2. A pipeline can be used from a Python script or notebook, allowing to run several classifiers at once on the same data sources
3. A pipeline can be run from the command line, in this case the pipeline configuration is defined in a JSON file

Usage examples can be found in the [usage notebook](https://gitlab.in2p3.fr/jbarnier/simulations-classifiers/-/blob/main/notebooks/usage.ipynb?ref_type=heads).

## Data preprocessing and classifiers

Each classifier take as input two folders of fasta files: one with empirical alignments, and one with simulated alignments.

These Fasta files are imported, and each sequence in each alignment is tokenized, _ie_ converted from a string to a sequence of integers.

If a sequence contains an ambiguous token (one of `"BZJUOX"` for AA, one of `"NDHVBRYKMSWX*"` for DNA), the entire sequence is removed from the alignement.

These tokenized alignments can then be used as input to one of the available classifiers:

| Classifier                     | Type                       | Data                               |
| ------------------------------ | -------------------------- | ---------------------------------- |
| `LogisticRegressionClassifier` | Logistic Regression        | `MsaCompositionData`               |
| `DenseMsaClassifier`           | Dense neural network       | `MsaCompositionData`               |
| `DenseSiteClassifier`          | Dense neural network       | `SiteCompositionData`              |
| `AACnnClassifier`              | CNN                        | `SiteCompositionData` for AA data  |
| `DNACnnClassifier`             | CNN                        | `SiteCompositionData` for DNA data |
| `AttentionClassifier`          | Transformer neural network | `SequencesData`                    |

The dataset type and corresponding data transformation depend on the classifier used. These are the available dataset types:

| Data type             | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `MsaCompositionData`  | Overall proportion of each AA or nucleotide per alignment  |
| `SiteCompositionData` | Per site proportion of each AA or nucleotide per alignment |
| `SequencesData`       | Sequence composition                                       |

## Repository structure

-   `config`: sample configuration file for classifiers pipeline
-   `data`: very small sample dataset of empirical and simulated amino acid alignments
-   `notebooks`: jupyter notebooks used for examples or tests
-   `src/classifiers`: source code of the Python `classifiers` package
    -   `src/classifiers/classif`: classifiers code and models
    -   `src/classifiers/data`: data sources, tokenizers and preprocessing
    -   `src/classifiers/metrics`: metrics definition and lists for deep learning classifiers
-   `tests`: Python package tests, can be run with `uv run pytest`

## Credits

Classifiers architecture and sample data set has been taken from the following paper:

> Johanna Trost, Julia Haag, Dimitri Höhler, Laurent Jacob, Alexandros Stamatakis, Bastien Boussau, Simulations of Sequence Evolution: How (Un)realistic They Are and Why, _Molecular Biology and Evolution_, Volume 41, Issue 1, January 2024, msad277, <https://doi.org/10.1093/molbev/msad277>
