"""
Dense network based classifiers.
"""

from classifiers.classif.deep_classifier import DeepClassifier
from classifiers.classif.models.dense_model import DenseMsaNet, DenseSiteNet
from classifiers.classif.torch_datasets import MsaDataset, SiteDataset
from classifiers.data.data import Data, MsaCompositionData, SiteCompositionData


class DenseSiteClassifier(DeepClassifier):
    """
    A dense neural network classifier for site composition data.

    Parameters
    ----------
    data : BaseData
        The input data for classification.
    max_width : int or None, optional
        The maximum width for the site composition data. Default is None.
    learning_rate : float, optional
        The learning rate for the optimizer. Default is 0.01.
    *args : tuple
        Additional positional arguments to be passed to the parent class.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.
    """

    def __init__(
        self, data: Data, max_width: int | None = None, learning_rate: float = 0.01, *args, **kwargs
    ):
        super().__init__(data, *args, learning_rate=learning_rate, **kwargs)

        self.max_width = max_width
        self.data = SiteCompositionData(data, max_width=self.max_width)
        self.dataset = SiteDataset(self.data.aligns, self.data.labels)

        self.n_sites, self.n_features = next(iter(self.data.aligns.values())).shape
        self.model = DenseSiteNet(n_features=self.n_features, n_sites=self.dataset.max_length)


class DenseMsaClassifier(DeepClassifier):
    """
    A dense neural network classifier for multiple sequence alignment (MSA) composition data.

    Parameters
    ----------
    data : BaseData
        The input data for classification.
    learning_rate : float, optional
        The learning rate for the optimizer. Default is 0.01.
    *args : tuple
        Additional positional arguments to be passed to the parent class.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.
    """

    def __init__(self, data: Data, learning_rate: float = 0.01, *args, **kwargs):
        super().__init__(data, *args, learning_rate=learning_rate, **kwargs)

        self.data = MsaCompositionData(data)
        self.dataset = MsaDataset(self.data.aligns, self.data.labels)

        self.n_features = next(iter(self.data.aligns.values())).shape[0]
        self.model = DenseMsaNet(n_features=self.n_features)
