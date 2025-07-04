from __future__ import annotations

import copy

from classifiers.data import preprocessing_fn
from classifiers.data.sources import DataSource
from classifiers.data.tokenizers import Tokenizer
from classifiers.logger import default_logger


class Data:
    """
    Base class for handling and preprocessing alignment data.

    Parameters
    ----------
    source_real : DataSource
        Source of real alignment data.
    source_simulated : DataSource
        Source of simulated alignment data.
    tokenizer : Tokenizer
        Tokenizer for processing the alignment data.
    """

    def __init__(self, source_real: DataSource, source_simulated: DataSource, tokenizer: Tokenizer) -> None:
        self.source_real = source_real
        self.source_simulated = source_simulated
        self.aligns = {}
        self.labels = {}
        self.tokenizer = tokenizer
        self.n_tokens = tokenizer.n_tokens + 1
        self.preprocess()

    def preprocess(self):
        """
        Preprocess the alignment data.

        This method applies common preprocessing to both real and simulated alignments.
        """
        real_aligns = self.source_real.aligns
        simulated_aligns = self.source_simulated.aligns
        self.aligns, self.labels = preprocessing_fn.common_preprocessing(
            real_aligns, simulated_aligns, self.tokenizer
        )

    def filter_aligns_by_ids(self, ids: list[str]) -> None:
        """
        Filter alignments and labels by removing specified IDs.

        Parameters
        ----------
        ids : list of str
            List of IDs to be removed from alignments and labels.

        Raises
        ------
        RuntimeError
            If no alignments are left after filtering.
        """
        default_logger.info(f"Removing {len(ids)}/{len(self.aligns)} alignments.")
        if len(ids) == len(self.aligns):
            msg = "No alignments left after filtering."
            raise RuntimeError(msg)
        self.aligns = {k: align for k, align in self.aligns.items() if k not in ids}
        self.labels = {k: label for k, label in self.labels.items() if k not in ids}

    def filter_aligns_width(self, width: int) -> None:
        """
        Filter alignments by width (number of sites).

        Alignments with width greater than the max width are removed from the dataset.

        Parameters
        ----------
        width : int
            Maximum allowed width for alignments.
        """
        default_logger.info(f"Limiting alignments width to {width}")
        ids = [k for k in self.aligns.keys() if self.aligns[k].shape[1] > width]
        self.filter_aligns_by_ids(ids)

    def filter_aligns_height(self, height: int) -> None:
        """
        Filter alignments by height (number of sequences).

        Alignments with height greater than the max height are removed from the dataset.

        Parameters
        ----------
        height : int
            Maximum allowed height for alignments.
        """
        default_logger.info(f"Limiting alignments height to {height}")
        ids = [k for k in self.aligns.keys() if self.aligns[k].shape[0] > height]
        self.filter_aligns_by_ids(ids)


class SequencesData:
    """
    Class for handling sequence data for classification.

    Parameters
    ----------
    data : Data
        Preprocessed base data.
    max_width : int, optional
        Maximum allowed width for alignments.
    max_height : int, optional
        Maximum allowed height for alignments.
    """

    def __init__(self, data: Data, max_width: int | None = None, max_height: int | None = None) -> None:
        self.data = copy.deepcopy(data)
        self.max_width, self.max_height = max_width, max_height

        if self.max_width is not None:
            self.data.filter_aligns_width(self.max_width)
        if self.max_height is not None:
            self.data.filter_aligns_height(self.max_height)

        self.aligns = data.aligns
        self.labels = data.labels


class MsaCompositionData:
    """
    Class for handling Multiple Sequence Alignment (MSA) composition data.

    Parameters
    ----------
    data : Data
        Preprocessed base data.
    """

    def __init__(self, data: Data) -> None:
        self.data = copy.deepcopy(data)
        self.aligns = preprocessing_fn.msa_composition_preprocessing(self.data.aligns, self.data.n_tokens)
        self.labels = self.data.labels


class SiteCompositionData:
    """
    Class for handling site composition data.

    Parameters
    ----------
    data : Data
        Preprocessed base data.
    max_width : int, optional
        Maximum allowed width for alignments.
    """

    def __init__(self, data: Data, max_width: int | None = None) -> None:
        self.data = copy.deepcopy(data)
        self.max_width = max_width

        if self.max_width is not None:
            self.data.filter_aligns_width(self.max_width)

        self.aligns = preprocessing_fn.site_composition_preprocessing(self.data.aligns, self.data.n_tokens)
        self.labels = self.data.labels
