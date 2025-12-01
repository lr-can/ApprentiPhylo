"""
Classes for handling sources of data.
"""

from itertools import chain
from pathlib import Path
from typing import Protocol

from classifiers.utils import StrAlignDict


class DataSource(Protocol):
    """
    Base protocol for data sources.
    """

    aligns: StrAlignDict


class FastaSource:
    """
    Data source for FASTA files.

    Parameters
    ----------
    data_path : Path | str
        Path to the directory containing FASTA files.
    """

    def __init__(self, data_path: Path | str) -> None:
        self.data_path = Path(data_path)

        if not self.data_path.is_dir():
            raise FileNotFoundError(f"{self.data_path} is not a valid directory.")

        # For consistency
        self.root = self.data_path

        # All files, sorted
        self.files = sorted(
            list(self.data_path.glob("*.fasta")) +
            list(self.data_path.glob("*.fa"))
        )

        # True filenames preserved
        self.filenames = [f.name for f in self.files]

        # Build aligns with TRUE filenames
        self.aligns = self.get_aligns()


    def get_aligns(self) -> StrAlignDict:
        """
        Load all FASTA files and return a dict:
            { "filename.fasta": [seq1, seq2, ...] }
        """
        fasta_files = chain(
            self.data_path.glob("*.fasta"),
            self.data_path.glob("*.fa")
        )

        aligns: StrAlignDict = {}
        for file in fasta_files:
            aligns[file.name] = FastaSource.parse_fasta(file)

        return aligns


    @staticmethod
    def parse_fasta(file: Path) -> list:
        """
        Parse a FASTA file and extract sequences.

        Returns
        -------
        list of sequences
        """
        sequences = []
        with open(file) as f:
            current_seq = []
            for line in f:
                if line.startswith(">"):
                    if len(current_seq) > 0:
                        sequences.append("".join(current_seq))
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if len(current_seq) > 0:
                sequences.append("".join(current_seq))

        return sequences


class DictSource:
    """
    Data source for a dictionary of alignments.
    """

    def __init__(
        self,
        aligns: StrAlignDict,
    ) -> None:
        self.aligns = aligns
