"""
CNN based classifiers.
"""

from classifiers.classif.deep_classifier import DeepClassifier
from classifiers.classif.models.cnn_models import AAConvNet, DNAConvNet
from classifiers.classif.torch_datasets import SiteDataset
from classifiers.data.data import Data, SiteCompositionData


class CnnClassifier(DeepClassifier):
    """
    Base Convolutional Neural Network (CNN) classifier.

    This class is not intended to be instantiated.

    Parameters
    ----------
    data : Data
        The input data for classification.
    max_width : int or None, optional
        The maximum width of the input data. Default is None.
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


class AACnnClassifier(CnnClassifier):
    """
    A CNN classifier for amino acid site composition data.

    Parameters
    ----------
    data : Data
        The input data for classification.
    max_width : int or None, optional
        The maximum width of the input data. Default is None.
    *args : tuple
        Additional positional arguments to be passed to the parent class.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    Attributes
    ----------
    model : AAConvNet
        The CNN model for amino acid sequence classification.
    """

    def __init__(self, data: Data, max_width: int | None = None, *args, **kwargs):
        kernel_size = kwargs.pop('kernel_size', 1)
        self.kernel_size = kernel_size
        super().__init__(data, max_width=max_width, *args, **kwargs)
        self.model = AAConvNet(n_features=self.n_features, n_sites=self.dataset.max_length, kernel_size=kernel_size)


class DNACnnClassifier(CnnClassifier):
    """
    A CNN classifier for DNA site composition data.

    Parameters
    ----------
    data : Data
        The input data for classification.
    max_width : int or None, optional
        The maximum width of the input data. Default is None.
    *args : tuple
        Additional positional arguments to be passed to the parent class.
    **kwargs : dict
        Additional keyword arguments to be passed to the parent class.

    Attributes
    ----------
    model : DNAConvNet
        The CNN model for DNA sequence classification.
    """

    def __init__(self, data: Data, max_width: int | None = None, *args, **kwargs):
        kernel_size = kwargs.pop('kernel_size', 3)
        self.kernel_size = kernel_size
        super().__init__(data, max_width=max_width, *args, **kwargs)

        self.model = DNAConvNet(n_features=self.n_features, n_sites=self.dataset.max_length, kernel_size=kernel_size)
