"""
phylo_metrics.py
=================
Calcule des métriques phylogénétiques à partir d’arbres au format Newick.  
Implémente notamment le MPD (Mean Pairwise Distance) et d’autres statistiques de structure d’arbre.  
Utilisé par report.py pour enrichir les rapports PDF.
"""

from Bio import Phylo
import numpy as np
from pathlib import Path

def mean_pairwise_distance(tree_file):
    """
    Calcule le MPD (Mean Pairwise Distance) pour un arbre Newick.
    tree_file : chemin vers le fichier .nw ou .nwk
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


def tree_summary(tree_file):
    """
    Retourne un dictionnaire de métriques phylogénétiques.
    """
    return {
        "MPD": mean_pairwise_distance(tree_file),
        "n_leaves": len(Phylo.read(tree_file, "newick").get_terminals())
    }
