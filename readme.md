# 🧬 PhyloClassifier - Automated pipeline for simulation and classification

Ce dépôt fournit un **pipeline bioinformatique complet** pour :

- **Simuler** des alignements phylogénétiques à partir de données réelles  
- **Reconstruire** des arbres phylogénétiques  
- **Calculer** des métriques phylo (MPD, n feuilles)  
- **Classifier** alignements réels vs simulés (Run1 / Run2)  
- **Générer** un **rapport PDF** complet  
- **Visualiser** les résultats dans un dashboard interactif

Toutes les étapes sont centralisées dans un seul script :

python3 scripts/main2.py <simulate|classify|visualisation>

---

# 📑 Table des matières

- [🎯 Objectifs](#-objectifs)
- [📦 Installation](#-installation)
- [📂 Structure](#-structure)
- [🚀 Utilisation](#-utilisation)
  - [1️⃣ Simulation](#1️⃣-simulation)
  - [2️⃣ Classification](#2️⃣-classification)
  - [📄 Rapport PDF](#-rapport-pdf)
  - [📊 Dashboard interactif](#-dashboard-interactif)
- [🧱 Détails techniques](#-détails-techniques)
- [🗂️ Logging & Reproductibilité](#️-logging--reproductibilité)
- [🧪 Exemples](#-exemples)
- [📬 Contact](#-contact)

---

# 🎯 Objectifs

Le pipeline combine :

✔ Prétraitement d’alignements réels  
✔ Simulation via **BppSuite**  
✔ Reconstruction d’arbres phylogénétiques  
✔ Calcul de métriques  
✔ Classification des alignements  
✔ Export PDF  
✔ Dashboard interactif (Dash)

---

# 📦 Installation

### Dépendances Python

pip install -r requirements.txt

yaml
Copier le code

### Outils externes requis

- **BppSuite** (`bppseqgen`, `bppml`, etc.)
- IQTree, FastTree, ou RAxML selon configuration
- **LaTeX** (optionnel, pour les PDF)

---

# 📂 Structure

.
├── scripts/
│ ├── main2.py # Pipeline principal
│ ├── preprocess.py
│ ├── simulation.py
│ ├── compute_tree.py
│ ├── phylo_metrics.py
│ ├── classification.py
│ ├── analyse_classif.py
│ ├── fix_logreg_history.py
│ ├── dashboard.py
│
├── data/
├── backup/
├── results/
├── logs/
└── README.md

yaml
Copier le code

---

# 🚀 Utilisation

## 1️⃣ Simulation

Effectue :

1. Prétraitement des alignements
2. Simulation (Bio++)
3. Reconstruction des arbres
4. Calcul des métriques MPD

### Commande

python3 scripts/main2.py simulate
--pre-input <dir_raw>
--pre-output <dir_clean>
--minseq N --maxsites N --minsites N
--alphabet aa|dna
--align <clean_dir>
--tree <tree_dir>
--config <model.bpp>
--sim-output <dir>
--ext_rate <float>
--tree-output <dir>
--metrics-output <dir>

yaml
Copier le code

---

## 2️⃣ Classification

Deux modes disponibles :

| Mode | Description |
|------|-------------|
| **Run1** | Classification simple |
| **Run1 + Run2** | Raffinement itératif |

### Commande

python3 scripts/main2.py classify
--real-align <dir>
--sim-align <dir>
--output <dir>
--config <file.json>
--tools <dir>
[--two-iterations]
[--threshold 0.5]
[--report-output report.pdf]

yaml
Copier le code

---

## 📄 Rapport PDF

Le PDF inclut :

- Résumé du modèle
- Performances (Run1 / Run2)
- Courbes logistic regression
- Tableaux récapitulatifs
- Diagnostics

Il est généré si `--report-output` est fourni.

---

## 📊 Dashboard interactif

Lancement :

python3 scripts/main2.py visualisation

markdown
Copier le code

Fonctionnalités :

- Visualisation des scores
- Comparaison des simulateurs
- Exploration des distances phylo
- Filtres dynamiques

---

# 🧱 Détails techniques

### 🔹 Prétraitement (`Preprocess`)
- Filtre séquences courtes
- Supprime gaps
- Supprime sites ambigus (stratégies `gapless` & `clean`)

### 🔹 Simulation (`BppSimulator`)
- Utilise `bppseqgen`
- Modèles configurables (`.bpp`)
- Taux d’extinction ajustable

### 🔹 Arbres (`ComputingTrees`)
- IQTree / FastTree selon outils disponibles
- Sortie en `.nwk`

### 🔹 Métriques (`tree_summary`)
- MPD
- Nombre de feuilles

### 🔹 Classification (`run_classification`)
- Logistic regression
- Réentraînement (Run2) optionnel
- Score threshold configurable

### 🔹 Rapport PDF
- Basé sur `analyse_classif.py`
- Figures intégrées
- Résumé analysé

---

# 🗂️ Logging & Reproductibilité

Chaque étape écrit dans :  
`logs/pipeline_log.csv`

Champs :

| Champ | Description |
|-------|-------------|
| `step` | Étape du pipeline |
| `status` | success / error |
| `duration` | Temps d'exécution |
| `args` | Paramètres exacts |

Permet une **auditabilité complète**.

---

# 🧪 Exemples

### ▶️ Simulation complète

python3 scripts/main2.py simulate
--pre-input data/prot_mammals
--pre-output results/preprocessed
--minseq 5 --maxsites 2000 --minsites 100
--alphabet aa
--align results/preprocessed/clean_data
--tree data/prot_mammals/trees
--config backup/config/bpp/aa/WAG_frequencies.bpp
--sim-output results/simulations
--ext_rate 0.3
--tree-output results/trees
--metrics-output results/metrics

shell
Copier le code

### ▶️ Classification simple

python3 scripts/main2.py classify
--real-align results/preprocessed/clean_data
--sim-align results/simulations
--output results/classification
--config backup/config_template.json
--tools backup/

shell
Copier le code

### ▶️ Classification + Run2 + PDF

python3 scripts/main2.py classify
--real-align results/preprocessed/clean_data
--sim-align results/simulations
--output results/classification
--config backup/config_template.json
--tools backup/
--two-iterations
--report-output results/classification/final_report.pdf

shell
Copier le code

### ▶️ Dashboard

python3 scripts/main2.py visualisation

yaml
Copier le code

---

# 📬 Contact

Pour questions, suggestions ou contributions :  
**<ton email / lien GitHub>**
