"""
AA and DNA sequence tokenizers.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from classifiers.utils import AMBIG_TOKEN


class Tokenizer:
    """
    A class for tokenizing AA or DNA sequences from strings to arrays of integers.

    Parameters
    ----------
    tokens : Sequence
        A sequence of valid tokens.
    ambig_tokens : Sequence
        A sequence of ambiguous tokens.

    Attributes
    ----------
    n_tokens : int
        Number of valid tokens.
    tokens_dict : dict
        Dictionary mapping tokens to their integer representations.
    ambig_tokens : set
        Set of ambiguous tokens.
    """

    def __init__(self, tokens: Sequence, ambig_tokens: Sequence):
        self.n_tokens = len(tokens)
        self.tokens_dict = {token: i + 1 for i, token in enumerate(tokens)}
        self.tokens_dict.update({token: AMBIG_TOKEN for token in ambig_tokens})
        self.ambig_tokens = set(ambig_tokens)

    def tokenize(self, sequence: str) -> NDArray[np.int8]:
        """
        Tokenize a given sequence.

        Parameters
        ----------
        sequence : str
            The input sequence to be tokenized.

        Returns
        -------
        NDArray[np.int8]
            An array of integers representing the input sequence.
        """
        res = np.array([self.tokens_dict[c] for c in sequence], dtype=np.int8)
        return res

    def has_ambig_site(self, sequence: np.ndarray) -> bool:
        """
        Check if the sequence contains any ambiguous tokens.

        Parameters
        ----------
        sequence : np.ndarray
            The tokenized sequence to check.

        Returns
        -------
        bool
            True if the sequence contains an ambiguous token, False otherwise.
        """
        return AMBIG_TOKEN in sequence


AA_TOKENIZER = Tokenizer("-ARNDCQEGHILKMFPSTWYV", ambig_tokens="BZJUOX")
DNA_TOKENIZER = Tokenizer("-ATGC", ambig_tokens="NDHVBRYKMSWX*")
