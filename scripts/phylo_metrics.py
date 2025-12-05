"""
phylo_metrics.py
=================
Calcule des métriques phylogénétiques à partir d’arbres au format Newick.  
Implémente notamment le MPD (Mean Pairwise Distance) et d’autres statistiques de structure d’arbre.  
Utilisé par report.py pour enrichir les rapports PDF.
"""

import os
import subprocess
from Bio import SeqIO, Phylo
from Bio.Phylo import read
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def mean_pairwise_distance(tree_file):
    """
    Calcule le MPD (Mean Pairwise Distance) pour un arbre Newick.
    tree_file : chemin vers le fichier .nw ou .nwk
    Args:
        tree_file (str): Chemin vers le fichier Newick de l’arbre.
    Returns:
        float: Valeur du MPD.
    """
    tree_file = Path(tree_file)
    tree = Phylo.read(tree_file, "newick")

    terminals = tree.get_terminals()
    n = len(terminals)
    if n < 2:
        return 0.0

    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dist = tree.distance(terminals[i], terminals[j])
            dists.append(dist)
    
    return np.mean(dists)


##### MPD with M. Gueguen's code
# ALIGNMENT + TREE FUNCTIONS 
def combine_alignments(empirical_file, simulation_file, output_file):
    records = []
    for record in SeqIO.parse(empirical_file, "fasta"):
        record.id = f"empirical_{record.id}"
        records.append(record)
    for record in SeqIO.parse(simulation_file, "fasta"):
        record.id = f"simulation_{record.id}"
        records.append(record)
    SeqIO.write(records, output_file, "fasta")


def generate_tree(alignment_file, tree_file):
    try:
        cmd = ["fasttree", "-lg", "-gamma", alignment_file]
        with open(tree_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, stderr=result.stderr)
    except Exception as e:
        logging.error(f"Tree generation failed: {e}")
        raise


def calculate_closest_distances(tree_file):
    try:
        tree = read(tree_file, "newick")
        leaves = tree.get_terminals()

        empirical = [l for l in leaves if l.name.startswith("empirical_")]
        simulation = [l for l in leaves if l.name.startswith("simulation_")]

        distances = []
        for sim in simulation:
            mind = float('inf')
            for emp in empirical:
                d = tree.distance(sim, emp)
                if d > 0:
                    mind = min(mind, d)
            if mind != float('inf'):
                distances.append(mind)
        return distances if distances else None

    except Exception as e:
        logging.error(f"Error computing distances: {e}")
        return None


# METRIC: MPD FUNCTION  #
def compute_mpd(distances):
    if not distances:
        return None
    return sum(distances) / len(distances)

def compute_metrics_for_pair(emp_file, sim_file, outdir):
    basename = os.path.basename(emp_file)

    res_dir = os.path.join(outdir, "metrics_results")
    os.makedirs(res_dir, exist_ok=True)

    aln_dir = os.path.join(res_dir, 'combine_aln')
    tree_dir = os.path.join(res_dir, 'combine_tree')
    os.makedirs(aln_dir, exist_ok=True)
    os.makedirs(tree_dir, exist_ok=True)

    combined = os.path.join(aln_dir, basename)
    tree_file = os.path.join(tree_dir, basename.replace(".fasta", ".tree"))

    # Combinaison et génération de l'arbre
    combine_alignments(emp_file, sim_file, combined)
    generate_tree(combined, tree_file)

    # Calcul des distances et MPD
    distances = calculate_closest_distances(tree_file)
    mpd = compute_mpd(distances)

    # --- AJOUT : calcul du nombre de feuilles ---
    tree = Phylo.read(tree_file, "newick")
    n_leaves = len(tree.get_terminals())

    # Retourner également n_leaves
    return basename, mpd, n_leaves
