"""
Pytorch Dataset subclasses for deep learning classifiers.
"""

import torch
from torch.utils.data import Dataset

from classifiers.utils import FloatAlignDict, IntAlignDict, LabelDict


class SiteDataset(Dataset):
    """
    Dataset for site composition data.

    Parameters
    ----------
    aligns : dict of {str: NDArray[np.float32]}
        Dictionary of alignments, where keys are identifiers and values are numpy arrays.
    labels : dict of {str: int}
        Dictionary of labels, where keys are identifiers and values are integer labels.
    """

    def __init__(self, aligns: FloatAlignDict, labels: LabelDict):
        self.aligns = aligns
        self.labels = labels
        self.keys = list(self.aligns.keys())
        self.max_length = max([align.shape[0] for align in self.aligns.values()])

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves an item from the dataset.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the alignment tensor and label tensor.
        """
        key = self.keys[index]
        align = torch.tensor(self.aligns[key], dtype=torch.float32)
        padding = (0, 0, 0, self.max_length - align.shape[0])  # (left, right, top, bottom) padding
        align = torch.nn.functional.pad(align, padding, mode="constant", value=0)
        label = torch.tensor(self.labels[key]).unsqueeze(-1).float()
        return align, label


class MsaDataset(Dataset):
    """
    Dataset for multiple sequence alignment (MSA) composition data.

    Parameters
    ----------
    aligns : dict of {str: NDArray[np.float32]}
        Dictionary of alignments, where keys are identifiers and values are numpy arrays.
    labels : dict of {str: int}
        Dictionary of labels, where keys are identifiers and values are integer labels.
    """

    def __init__(self, aligns: FloatAlignDict, labels: LabelDict):
        self.aligns = aligns
        self.labels = labels
        self.keys = list(self.aligns.keys())

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves an item from the dataset.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the alignment tensor and label tensor.
        """
        key = self.keys[index]
        align = torch.tensor(self.aligns[key], dtype=torch.float32)
        label = torch.tensor(self.labels[key]).unsqueeze(-1).float()
        return align, label


class SequencesDataset(Dataset):
    """
    Dataset for sequence data.

    Parameters
    ----------
    aligns : dict of {str: NDArray[np.int8]}
        Dictionary of alignments, where keys are identifiers and values are numpy arrays.
    labels : dict of {str: int}
        Dictionary of labels, where keys are identifiers and values are integer labels.
    """

    def __init__(self, aligns: IntAlignDict, labels: LabelDict):
        self.aligns = aligns
        self.labels = labels
        self.keys = list(self.aligns.keys())

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves an item from the dataset.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            A tuple containing the alignment tensor and label tensor.
        """
        key = self.keys[index]
        align = torch.tensor(self.aligns[key], dtype=torch.int64)
        label = torch.tensor(self.labels[key]).unsqueeze(-1).float()
        return align, label
