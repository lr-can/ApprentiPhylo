"""
Tests for features computation.
"""

import pytest

from classifiers.data.tokenizers import AA_TOKENIZER, DNA_TOKENIZER
from classifiers.utils import AMBIG_TOKEN


def test_aa_tokenizer():
    seq = "-ARNDCQEGHILKMFPSTWYV"
    assert list(AA_TOKENIZER.tokenize(seq)) == [i + 1 for i in range(len(seq))]
    seq = "--AAVV-BZJUOX"
    assert list(AA_TOKENIZER.tokenize(seq)) == [1, 1, 2, 2, 21, 21, 1] + [AMBIG_TOKEN] * 6

    with pytest.raises(KeyError):
        AA_TOKENIZER.tokenize("ARN4")


def test_dna_tokenizer():
    seq = "-ATGC"
    assert list(DNA_TOKENIZER.tokenize(seq)) == [i + 1 for i in range(len(seq))]
    seq = "--AACC-NDHVBRYKMSWX*"
    assert list(DNA_TOKENIZER.tokenize(seq)) == [1, 1, 2, 2, 5, 5, 1] + [AMBIG_TOKEN] * 13
    with pytest.raises(KeyError):
        AA_TOKENIZER.tokenize("ATG2")
