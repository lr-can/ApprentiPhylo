"""
CNN based classifiers.
"""

from classifiers.classif.deep_classifier import DeepClassifier
from classifiers.classif.models.cnn_models import AAConvNet, DNAConvNet
from classifiers.classif.torch_datasets import SiteDataset
from classifiers.data.data import Data
from classifiers.data import preprocessing_fn


class CnnClassifier(DeepClassifier):
    """
    Base CNN classifier.
    NOTE: We NO LONGER replace self.data with SiteCompositionData.
          Instead, we compute site-composition matrices ON THE FLY
          and build self.dataset accordingly.

          This keeps DeepClassifier.predict() WORKING:
              -> self.data is still a Data()
              -> self.data.dataset exists
    """

    def __init__(
        self, data: Data, max_width: int | None = None, learning_rate: float = 0.01, *args, **kwargs
    ):
        super().__init__(data, *args, learning_rate=learning_rate, **kwargs)

        self.max_width = max_width

        # ---------------------------------------------------------------------
        # 1) On clone les alignements de Data avant transformation
        # ---------------------------------------------------------------------
        aligns = data.aligns
        labels = data.labels

        # 2) On applique la transformation site-composition (ancienne SiteCompositionData)
        site_aligns = preprocessing_fn.site_composition_preprocessing(
            aligns,
            data.n_tokens,
        )

        # 3) On filtre en width si demandé
        if self.max_width is not None:
            site_aligns = {
                k: v for k, v in site_aligns.items()
                if v.shape[0] <= self.max_width
            }
            labels = {k: labels[k] for k in site_aligns.keys()}

        # 4) Construction du vrai dataset CNN
        self.dataset = SiteDataset(site_aligns, labels)

        # 5) On expose les infos pour les modèles
        any_align = next(iter(site_aligns.values()))
        self.n_sites, self.n_features = any_align.shape

        # DeepClassifier.train() utilisera self.dataset correctement
        # DeepClassifier.predict() ne l’utilise PAS → ok


class AACnnClassifier(CnnClassifier):
    """
    CNN for amino acid composition.
    """

    def __init__(self, data: Data, max_width: int | None = None, *args, **kwargs):
        kernel_size = kwargs.pop("kernel_size", 1)
        self.kernel_size = kernel_size

        super().__init__(data, max_width=max_width, *args, **kwargs)

        self.model = AAConvNet(
            n_features=self.n_features,
            n_sites=self.dataset.max_length,
            kernel_size=kernel_size,
        )


class DNACnnClassifier(CnnClassifier):
    """
    CNN for DNA composition.
    """

    def __init__(self, data: Data, max_width: int | None = None, *args, **kwargs):
        kernel_size = kwargs.pop("kernel_size", 3)
        self.kernel_size = kernel_size

        super().__init__(data, max_width=max_width, *args, **kwargs)

        self.model = DNAConvNet(
            n_features=self.n_features,
            n_sites=self.dataset.max_length,
            kernel_size=kernel_size,
        )
