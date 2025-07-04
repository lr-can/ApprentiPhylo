"""
Attention based classifiers.
"""

from classifiers.classif.deep_classifier import DeepClassifier
from classifiers.classif.models.attention_models import AttentionNet
from classifiers.classif.torch_datasets import SequencesDataset
from classifiers.data.data import Data, SequencesData


class AttentionClassifier(DeepClassifier):
    """
    Attention-based classifier for sequence data.

    This classifier uses an attention mechanism to process and classify sequence data.

    Parameters
    ----------
    data : Data
        The input data for classification.
    max_width : int, optional
        Maximum width for the sequence data. Default is None.
    max_height : int, optional
        Maximum height for the sequence data. Default is None.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.001.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(
        self,
        data: Data,
        max_width: int | None = None,
        max_height: int | None = None,
        learning_rate: float = 0.001,
        *args,
        **kwargs,
    ):
        super().__init__(data, *args, batch_pad_sequences=True, learning_rate=learning_rate, **kwargs)
        self.data = SequencesData(data, max_width=max_width, max_height=max_height)
        self.dataset = SequencesDataset(self.data.aligns, self.data.labels)

        self.n_sites, self.n_features = next(iter(self.data.aligns.values())).shape
        self.model = AttentionNet(n_features=self.n_features)
