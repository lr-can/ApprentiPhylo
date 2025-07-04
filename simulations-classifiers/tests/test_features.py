"""
Tests for features computation.
"""

import numpy as np
import pytest

from classifiers.data.features_fn import (
    n_invariant_sites,
    n_patterns,
    n_sites,
    n_taxa,
    sites_over_taxa,
)


@pytest.fixture
def sample_alignment():
    return np.array([[0, 1, 2, 3, 0], [0, 1, 2, 0, 0], [0, 1, 3, 3, 0], [0, 1, 2, 3, 0]])


def test_n_sites(sample_alignment):
    assert n_sites(sample_alignment) == 5


def test_n_taxa(sample_alignment):
    assert n_taxa(sample_alignment) == 4


def test_sites_over_taxa(sample_alignment):
    assert sites_over_taxa(sample_alignment) == 1.25


def test_n_invariant_sites(sample_alignment):
    assert n_invariant_sites(sample_alignment) == 3


def test_n_patterns(sample_alignment):
    assert n_patterns(sample_alignment) == 4


def test_empty_alignment():
    empty_align = np.array([[]])
    assert n_sites(empty_align) == 0
    assert n_taxa(empty_align) == 1
    assert sites_over_taxa(empty_align) == 0
    assert n_invariant_sites(empty_align) == 0
    assert n_patterns(empty_align) == 0


def test_single_sequence_alignment():
    single_seq_align = np.array([[0, 1, 2, 3]])
    assert n_sites(single_seq_align) == 4
    assert n_taxa(single_seq_align) == 1
    assert sites_over_taxa(single_seq_align) == 4.0
    assert n_invariant_sites(single_seq_align) == 4
    assert n_patterns(single_seq_align) == 4


def test_all_invariant_sites():
    invariant_align = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert n_invariant_sites(invariant_align) == 3
    assert n_patterns(invariant_align) == 1
