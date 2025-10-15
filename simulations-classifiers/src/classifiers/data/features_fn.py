"""
(OBSOLETE) Features computation.
"""

import numpy as np


def n_sites(align: np.ndarray) -> int:
    return align.shape[1]


def n_taxa(align: np.ndarray) -> int:
    return align.shape[0]


def sites_over_taxa(align: np.ndarray) -> float:
    return n_sites(align) / n_taxa(align)


def n_invariant_sites(align: np.ndarray) -> int:
    return np.sum(np.all(align == align[0], axis=0))


def n_patterns(align: np.ndarray) -> int:
    return np.unique(align, axis=1).shape[1]
