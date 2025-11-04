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

    Raises
    ------
    FileNotFoundError
        If the provided data_path is not a valid directory.
    """

    def __init__(self, data_path: Path | str) -> None:
        self.data_path = Path(data_path)
        self.aligns = self.get_aligns()

    def get_aligns(self) -> StrAlignDict:
        """
        Create self.aligns alignments dictionary from fasta files
        in self.data_path.
        """
        if not self.data_path.is_dir():
            msg = f"{self.data_path} is not a valid directory."
            raise FileNotFoundError(msg)

        # Import data from fasta files
        fasta_files = chain(self.data_path.glob("*.fasta"), self.data_path.glob("*.fa"))
        aligns = {f.stem: FastaSource.parse_fasta(f) for f in fasta_files}
        return aligns

    @staticmethod
    def parse_fasta(file: Path) -> list:
        """
        Parse a FASTA file and extract sequences.

        Parameters
        ----------
        file : Path
            Path to the FASTA file to be parsed.
        Returns
        -------
        list
            A list of sequences extracted from the FASTA file.
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

    Parameters
    ----------
    aligns : StrAlignDict
        Dictionary of alignments.
    tokenizer : BaseTokenizer
        The tokenizer used to process sequences.
    """

    def __init__(
        self,
        aligns: StrAlignDict,
    ) -> None:
        self.aligns = aligns
