# Scripts use.

## `alignment_description.py`

This script analyzes multiple sequence alignments. It produces histograms of the distributions of identity percentages, gaps, alignment lengths, and number of sequences per alignment on the alignment files. The script also generates a summary file containing this information.

**Parameters :**

* `--input` or `-i` : specifies the path of the directory containing the alignment files.
* `--output` or `-o` : specifies the path of the output directory where will be saved :
  * the summary file `data.tsv` with alignment statistics.
  * `distribution_identite.png` : Distribution of identity percentages.
  * `distribution_gaps.png` : Gaps percentage distribution.
  * `distribution_lengths.png` : Distribution of alignment lengths.
  * `distribution_sequences.png :` Distribution of number of sequences per alignment.

## `classification.py`

The `classification.py` script is used to configure and run an alignment classification pipeline. It adjusts an existing configuration file, integrating the paths to real alignments, simulated alignments and the output directory. It then executes a classification pipeline by calling an external script named pipeline.py from the simulations-classifiers directory of gitlab: https://gitlab.in2p3.fr/jbarnier/simulations-classifiers.

**Parameters :**

* `--realali` : path to the directory containing the real alignments to be classified.
* `--simali` : path to the directory containing the simulated alignments to be classified.
* `--output` : path to directory where results and updated configuration will be stored.
* `--config` : path to JSON configuration file containing initial parameters for classification pipeline. For example in : `config/classifiers/sample_conf_dna.json`
* `--tools` : path to the directory containing the a clone of the gitlab : https://gitlab.in2p3.fr/jbarnier/simulations-classifiers.

## `Compute_tree.py`

This script generates phylogenetic trees from alignment files using the FastTree tool. It process all alignment files in a given directory, or a specific subset if specified. The generated trees are saved in Newick format.

**Parameters :**

* `--input` or `-i` : path to directory containing input alignment files.
* `--output` or `-o` : path to directory where generated phylogenetic trees will be saved.
* `--alphabet` or `-a` : specifies the type of sequence used in alignments.
  * 'nt' : for nucleotide sequences. For nucleotide sequences, GTR model is used.
  * 'aa' : for protein sequences. For protein sequences, LG model is used.
* `--only` (optional) : path to a file listing the names of specific alignment files to be processed (one per line). If the option is supplied, only these files wille be taken into account.

## `Mapping.py`

This script enable mapping between alignments and phylogenetic trees to estimate the parameters of evolutionary models, notably the GTR and HKY models. Results are saved as files containing the estimated paramters for each alignment.

**Parameters :**

* `--align_path` : path to directory containing alignments (FASTA or other format).
* `--tree_path` : path to the directory containing the corresponding phylogenetic trees (newick format).
* `--config` : configuration file used by mapnh (from Bio++) to execute mapping. For example, `config/bpp/mapping/mapnh.bpp.`
* `--output` : Path to the directory where intermediate data ans estimated models will be stored.
  * `mapping_data` directory : intermediate file containing transition/transversions counts.
  * `mapping_GTR` directory : text files with GTR model parameters for each alignment.
  * `mapping_HKY` directory : text files with HKY template parameters for each alignment.

## `preprocessing_dataset.py`

This script is used to prepare a dataset of multiple sequence alignements for simulations. It performs cleaning steps, such as

* Deleting alignments with fewer sequences than a specified threshold.
* Deleting alignment too long or too short.
* Deleting site with site with ambiguous base or with gap.
  * For DNA : ambiguous sites are `NDHVBRYKMSWX*`.
  * For proteins : ambiguous sites are : `BZJUOX`.

**Parameters:**

* `--input` or `-i` : path to output directory where results will be saved.
* `--output` or `-o` : path to output direcory where results will be saved.
  * `clean_data` directory : Alignements that have passed initial filtering (number of sites between minsites and maxsites and containing at least minseq sequences)
  * `gap_less` : alignments that have passed initial filtering and without sites containing gaps.
  * `gap_and_ambigless` : alignments without sites containing gaps or ambiguous characters.
  * `ambigless` : alignments without sites containing ambiguous characters (directly after initial filtering).
  * `prepreocess.log` : log detailing pre-processing steps and associatied statistics.
* `--minseq` or `-s` : Minimum number of sequences required to keep the alignment.
* `--maxsites` : Maximum number of sites.
* `--minsites` : Minimum number of sites.
* `--type `: Type of sequences in alignments.
  * `aa` for proteins
  * `dna` for DNA

## `simulation.py`

This script can be used to simulate sequence alignments using different simulators : ESM or bppseqgen.

**Parameters :**

* `--simulator` or `-s` : List of simulators to be used ('ESM' or 'BPP')
* `--output `or `-o` : Output directory for simulated alignments.
* `--tree `or `-t` : path to directory containing phylogenetic trees.
* `--external_branch_length` or -e (optional for BPP): length of external branches to be applied
* `--align` or `-a` : directory containing alignments to be used as references.
* `--tools` : path to necessary tools, such as ESM scripts of Apptainer files (.cif).
* `--config` or `-c` (Optional for BPP): List of configuration files for BPP simulations. For example, config/bpp/dna/classic/JC69.bpp.
* `--modelmapping `or `-m` (optional for BPP): Path to the directory containing the evolution models to be applied
* `--gap` : Option to add gaps to simulated alignments (False or True). Default : False.

**Outputs :**

* `ESM` directory : alignments simulated with ESM simulator.
* `BPP` directory : alignments simulated with BPP, organized according to the configurations used.
* (Optionnal) `BPP_gap` directory : alignments with gaps.
* (Optionnal) `new_trees` directory : Phylogenetic trees modified if a specific external length has been defined.

## `taux_gc.py`

This script classifies multiple sequences alignments according to their average GC content. Alignments are separated into two groups : those with a high GC content and those with low content, according to a user-defined threshold.

**Parameters :**

* `--input` or `-i` : path to directory containing alignment files.
* `--output` or `-o` : path to directory where classified results will be saved.
* `--thresohold` or `-t` : threshold for average GC rate. alignment with a rate greater than or equal to this threshold will be classified in the high GC rate group.
