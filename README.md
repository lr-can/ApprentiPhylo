# Assessing the Realism of Phylogenetic Simulation Methods through Machine Learning

**Master’s Degree in Bioinformatics — Year 2**  
**University Claude Bernard Lyon 1**
**Project 12**

**Project team:** Lorcan Brenders, Thomas Gagnieu, Maya Givre  
**Supervisors:** Laurent Gueguen, Philippe Veber  
**Academic Year:** 2025–2026

---

## 🧬 Project Overview

This project aims to transform the existing *PhyloClassifier* repository into an automated, flexible, and reproducible pipeline for evaluating and improving the realism of phylogenetic sequence simulations using **machine learning**.

It integrates modules for:
- Simulation of protein alignments with **Bio++**
- Phylogenetic inference with **FastTree**
- Machine learning classification with **PyTorch**
- Quantitative metrics and visualisations (e.g. MPD, confusion matrices, ROC curves)

---

## 🚀 Development Phases

### 1. Basic Integration
Set up a minimal version of the pipeline capable of:
- Generating **phylogenetic trees** from real alignments using *FastTree*  
- Simulating **protein alignments** via *Bio++ (bppseqgen)*  
- Managing **input/output** files and configuration through `.bpp` templates  

🎯 *Objective:* Obtain a first functional version that produces simulated alignments and trees — a base for all subsequent phases.

---

### 2. Classification and Metrics
Integrate **PyTorch classifiers** and **metrics computation**:
- Implement logistic regression, simple MLP, and CNN classifiers  
- Compute quantitative metrics such as **Mean Pairwise Distance (MPD)** and **alignment variability**  
- Export results as `.csv` files  

🎯 *Objective:* Automatically compare real and simulated datasets and evaluate realism.

---

### 3. Visualisations
Add automatic generation of graphical outputs:
- Confusion matrices  
- Heatmaps  
- Distance distributions and tree plots  

All visualisations are exported as `.svg` and/or `.pdf` files.  

🎯 *Objective:* Provide interpretable graphical summaries of classifier and metric results.

---

### 4. Interactive Web Application *(optional)*
Develop an interactive **Dash**-based web interface:
- Explore classifier results and metrics dynamically  
- Integrate **Plotly** figures for real-time data visualisation  

🎯 *Objective:* Offer an intuitive graphical environment for users (biologists & bioinformaticians).

---

### 5. CLI and Documentation
Finalise a unified **Command-Line Interface (CLI)** and documentation:
- One main entry point to run the entire workflow (`--eval`, `--simu`, etc.)  
- User-friendly help messages and usage examples  
- Complete technical and user documentation with tutorials and test datasets  

🎯 *Objective:* Ensure full reproducibility, accessibility, and ease of use.

---

## 🧱 Deliverables and Milestones

### ✅ 1. Clean and Structured Repository
- Standard folder architecture:  
  ```
  /src
  /config
  /data
  /results
  /docs
  /scripts
  ```
- Example configuration files (`.bpp`, `.yaml`)  
- Minimal runnable pipeline script for local execution  

---

### ✅ 2. Functional Evaluation Mode
Command example:
```bash
phyloclf --eval --real <dir> --simu <dir> --out results/eval/
```
Generates:
- Metrics (`.csv`)
- Confusion matrices (`.svg`, `.pdf`)
- Logs and random seeds  

🎯 Validates complete data flow from input alignments to evaluation outputs.

---

### ✅ 3. Integration of MPD and FastTree
Implements **Mean Pairwise Distance (MPD)** computation:
- Builds trees with *FastTree*  
- Calculates patristic distances and inter-group MPD  
- Aggregates statistics with bootstrap/permutation testing  

Outputs:
- `mpd.csv` summary  
- Visualisations (histograms, violin/box plots)  
- Logs and parameters  

🎯 Adds biological interpretability and quantifies evolutionary realism.

---

### ✅ 4. Simulation and Analysis Mode
Adds simulation capabilities with *Bio++ bppseqgen*:
- Generates new alignments from real datasets  
- Automatically evaluates realism via classifiers  
- Optionally filters “realistic” alignments or optimises simulations  

🎯 Makes the pipeline autonomous for both **simulation** and **evaluation**.

---

### ✅ 5. Assessment of Simulation Variability
Computes diversity indicators:
- Site entropy  
- Mean Hamming distance  
- Amino acid frequency variance  

🎯 Detects over-deterministic simulators and ensures biological variability.

---

### ✅ 6. Visualisation and Dimensionality Reduction
Integrates **UMAP** and **MDS** projections to visualise:
- Proximity between simulated and real datasets  
- Clustering patterns reflecting simulation realism  

🎯 Helps interpret and diagnose simulation quality.

---

### ✅ 7. High-Performance and Reproducible Execution
- Local and SLURM cluster execution  
- Parallelisation via `--threads`  
- Containerisation (Docker / Apptainer)  
- Full reproducibility (identical outputs, hashes, and logs)

🎯 Guarantees scalability and environment consistency.

---

### ✅ 8. Testing, Documentation, and Reporting
- Continuous integration (CI) with unit and end-to-end tests  
- Comprehensive user guide and tutorial datasets  
- Final report summarising results, best classifier, and validation outcomes  

🎯 Ensures robustness, transparency, and long-term usability.

---

## 🧩 Validation Summary

Validation covers:
- **Technical:** reproducibility, stability, performance benchmarks  
- **Scientific:** classifier accuracy, MPD similarity, biological realism  
- **Usability:** intuitive CLI and documentation clarity  
- **Cross-platform:** local, containerised, and SLURM runs  

---

## 🧠 Expected Outputs
From any valid FASTA input, the pipeline produces:
- Simulated alignments (FASTA)
- Phylogenetic trees (Newick)
- MPD and variability metrics (CSV)
- Classifier reports (CSV, confusion matrices)
- Visualisations (SVG/PDF)
- Execution logs and version trace

---

## 🧾 Success Criteria
- End-to-end execution without errors  
- ≥0.80 F1-score for classifiers  
- Full reproducibility (100% identical outputs)  
- Processing time <10 minutes for 10 alignments (8 cores, 16 GB RAM)  
- 80%+ success rate in tutorial reproduction by external testers  