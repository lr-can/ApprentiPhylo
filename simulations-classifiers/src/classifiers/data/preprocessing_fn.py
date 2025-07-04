"""
Alignment data preprocessing functions.
"""

from itertools import zip_longest

import numpy as np
from numpy.typing import NDArray

from classifiers.data.tokenizers import Tokenizer
from classifiers.logger import default_logger
from classifiers.utils import (
    AMBIG_TOKEN,
    LABEL_REAL,
    LABEL_SIMULATED,
    PADDING_TOKEN,
    FloatAlignDict,
    IntAlignDict,
    LabelDict,
    StrAlignDict,
)


def site_composition(align: NDArray[np.int8], n_values: int) -> NDArray[np.float32]:
    """
    Compute site composition proportions for an alignment.

    Parameters
    ----------
    align : NDArray[np.int8]
        The alignment array.
    n_values : int
        The number of possible values.

    Returns
    -------
    NDArray[np.float32]
        An array of site composition proportions.
    """
    counts = np.apply_along_axis(np.bincount, axis=0, arr=align, minlength=n_values)
    counts = counts / counts.sum(axis=0, keepdims=True)
    return counts.T


def msa_composition(align: NDArray[np.int8], n_values: int) -> NDArray[np.float32]:
    """
    Compute overall composition frequencies for an alignment.

    Parameters
    ----------
    align : NDArray[np.int8]
        The alignment array.
    n_values : int
        The number of possible values.

    Returns
    -------
    NDArray[np.float32]
        An array of overall composition frequencies.
    """
    counts = np.bincount(align.flatten(), minlength=n_values)
    counts = counts / counts.sum()
    return counts


def n_seq(aligns: dict[str, NDArray]) -> int:
    """
    Count the total number of sequences across all alignments in a dictionary.

    Parameters
    ----------
    aligns : dict[str, NDArray]
        A dictionary of alignments.

    Returns
    -------
    int
        The total number of sequences.
    """
    return sum([align.shape[0] for align in aligns.values()])


def tokenize_aligns(aligns: StrAlignDict, tokenizer: Tokenizer) -> IntAlignDict:
    """
    Tokenize a dictionary of alignments.

    The tokenizer is used to convert each sequence in each alignment from a string to
    a list of integers.

    Parameters
    ----------
    aligns : StrAlignDict
        A dictionary of alignments, where each alignment is a list of string sequences.
    tokenizer : Tokenizer
        The tokenizer to use.

    Returns
    -------
    IntAlignDict
        A dictionary of tokenized alignments.
    """
    tokenized_aligns = {}
    for id, align in aligns.items():
        tokenized_align = [tokenizer.tokenize(seq) for seq in align]
        tokenized_align = list(zip_longest(*tokenized_align, fillvalue=PADDING_TOKEN))
        tokenized_aligns[id] = np.array(tokenized_align).T
    return tokenized_aligns


def filter_ambig_align(align: NDArray[np.int8]) -> tuple[NDArray[np.int8], int]:
    """
    Filter out sequences with ambiguous tokens from an alignment.

    Sequences with at least one ambiguous token are removed from the alignment.

    Parameters
    ----------
    align : NDArray[np.int8]
        The alignment matrix.

    Returns
    -------
    tuple[NDArray[np.int8], int]
        The filtered alignment and the number of removed sequences.
    """
    n_seq_before = align.shape[0]
    mask = np.any(align == AMBIG_TOKEN, axis=1)
    align = align[~mask]
    n_ambig = n_seq_before - align.shape[0]
    return align, n_ambig


def filter_ambig(aligns: IntAlignDict) -> IntAlignDict:
    """
    Filter out sequences with ambiguous tokens from a dictionary of alignments.

    Parameters
    ----------
    aligns : IntAlignDict
        A dictionary of alignments.

    Returns
    -------
    IntAlignDict
        A dictionary of filtered alignments.
    """
    n_seq_total = n_seq(aligns)
    n_aligns = len(aligns)
    n_seq_ambig = 0
    n_aligns_ambig = 0

    for k, align in aligns.items():
        filtered_align, n_ambig = filter_ambig_align(align)
        aligns[k] = filtered_align
        n_aligns_ambig += n_ambig > 0
        n_seq_ambig += n_ambig
    msg = (
        f"{n_seq_ambig}/{n_seq_total} sequences have been removed from "
        f"{n_aligns_ambig}/{n_aligns} alignements due to ambiguous sites."
    )
    default_logger.info(msg)
    return aligns


def common_preprocessing(
    source_aligns_real: StrAlignDict,
    source_aligns_simulated: StrAlignDict,
    tokenizer: Tokenizer,
) -> tuple[IntAlignDict, LabelDict]:
    """
    Perform common preprocessing steps on real and simulated alignments.

    Parameters
    ----------
    source_aligns_real : StrAlignDict
        A dictionary of real alignments.
    source_aligns_simulated : StrAlignDict
        A dictionary of simulated alignments.
    tokenizer : Tokenizer
        The tokenizer to use.

    Returns
    -------
    tuple[IntAlignDict, LabelDict]
        A tuple containing the preprocessed alignments and their labels.
    """
    default_logger.info("Tokenizing real alignments...")
    aligns_real = tokenize_aligns(source_aligns_real, tokenizer=tokenizer)
    default_logger.info("Removing sequences with ambiguous tokens...")
    aligns_real = filter_ambig(aligns_real)

    default_logger.info("Tokenizing simulated alignments...")
    aligns_simulated = tokenize_aligns(source_aligns_simulated, tokenizer=tokenizer)
    default_logger.info("Removing sequences with ambiguous tokens...")
    aligns_simulated = filter_ambig(aligns_simulated)

    default_logger.info("Checking alignments...")
    duplicated_keys = set(aligns_real.keys()) & set(aligns_simulated.keys())
    if len(duplicated_keys) > 0:
        msg = f"{len(duplicated_keys)} renamed keys in simulated aligns."
        default_logger.info(msg)
        for key in duplicated_keys:
            new_key = f"{key}_sim"
            aligns_simulated[new_key] = aligns_simulated.pop(key)

    default_logger.info("Creating labels...")
    labels_real = {k: LABEL_REAL for k in aligns_real.keys()}
    labels_simulated = {k: LABEL_SIMULATED for k in aligns_simulated.keys()}

    default_logger.info("Merging aligns and labels...")
    aligns = aligns_real | aligns_simulated
    labels = labels_real | labels_simulated

    default_logger.info(f"Total number of alignments: {len(aligns)}")

    return aligns, labels


def msa_composition_preprocessing(aligns: IntAlignDict, n_tokens: int) -> FloatAlignDict:
    """
    Compute MSA compositions for a dictionary of alignments.

    Parameters
    ----------
    aligns : IntAlignDict
        A dictionary of alignments.
    n_tokens : int
        The number of possible tokens.

    Returns
    -------
    FloatAlignDict
        A dictionary of MSA compositions.
    """
    msa_aligns = {}
    for k, align in aligns.items():
        msa_aligns[k] = msa_composition(align, n_values=n_tokens)
    return msa_aligns


def site_composition_preprocessing(aligns: IntAlignDict, n_tokens: int) -> FloatAlignDict:
    """
    Compute site compositions for a dictionary of alignments.

    Parameters
    ----------
    aligns : IntAlignDict
        A dictionary of alignments.
    n_tokens : int
        The number of possible tokens.

    Returns
    -------
    FloatAlignDict
        A dictionary of MSA compositions.
    """

    site_aligns = {}
    for k, align in aligns.items():
        align = site_composition(align, n_values=n_tokens)
        # align = torch.tensor(align)  # type: ignore
        site_aligns[k] = align

    return site_aligns
