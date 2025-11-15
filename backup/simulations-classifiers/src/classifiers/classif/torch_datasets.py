"""
Pytorch Dataset subclasses for deep learning classifiers.
"""

import torch
from torch.utils.data import Dataset

from classifiers.utils import FloatAlignDict, IntAlignDict, LabelDict


class SiteDataset(Dataset):
    """
    Dataset for site composition data.

    Returns (align, label, filename).
    """

    def __init__(self, aligns: FloatAlignDict, labels: LabelDict):
        self.aligns = aligns
        self.labels = labels
        self.keys = list(self.aligns.keys())
        self.max_length = max([align.shape[0] for align in self.aligns.values()])

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple:
        key = self.keys[index]
        align = torch.tensor(self.aligns[key], dtype=torch.float32)

        # pad (seq_length → max_length)
        padding = (0, 0, 0, self.max_length - align.shape[0])
        align = torch.nn.functional.pad(align, padding, mode="constant", value=0)

        label = torch.tensor(self.labels[key]).unsqueeze(-1).float()

        return align, label, key   #  <-- FIX 🔥


class MsaDataset(Dataset):
    """
    Dataset for MSA composition data.

    Returns (align, label, filename).
    """

    def __init__(self, aligns: FloatAlignDict, labels: LabelDict):
        self.aligns = aligns
        self.labels = labels
        self.keys = list(self.aligns.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple:
        key = self.keys[index]
        align = torch.tensor(self.aligns[key], dtype=torch.float32)
        label = torch.tensor(self.labels[key]).unsqueeze(-1).float()

        return align, label, key


class SequencesDataset(Dataset):
    """
    Dataset for raw integer-encoded sequences.

    Returns (align, label, filename).
    """

    def __init__(self, aligns: IntAlignDict, labels: LabelDict):
        self.aligns = aligns
        self.labels = labels
        self.keys = list(self.aligns.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> tuple:
        key = self.keys[index]
        align = torch.tensor(self.aligns[key], dtype=torch.int64)
        label = torch.tensor(self.labels[key]).unsqueeze(-1).float()

        return align, label, key 
