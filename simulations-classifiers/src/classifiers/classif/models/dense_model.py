"""
Models used by dense network classifiers.
"""

import torch
from torch import nn


class DenseSiteNet(nn.Module):
    """
    A dense neural network for site-based classification.

    Parameters
    ----------
    n_features : int
        Number of features per site.
    n_sites : int
        Number of sites.
    """

    def __init__(
        self,
        n_features: int,
        n_sites: int,
    ):
        super().__init__()
        self.dense_layer1 = nn.Sequential(nn.Linear(n_sites * n_features, 100), nn.ReLU(), nn.Dropout(0.2))
        self.dense_layer2 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_sites, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        # x is (B, S, C)
        x = x.view(x.shape[0], -1)
        # -> x is (B, S * C)
        x = self.dense_layer1(x)
        # -> x is (B, 100)
        out = self.dense_layer2(x)
        # -> out is (B, 1)

        return out


class DenseMsaNet(nn.Module):
    """
    A dense neural network for MSA-based classification.

    Parameters
    ----------
    n_features : int
        Number of features.
    """

    def __init__(
        self,
        n_features: int,
    ):
        super().__init__()
        self.dense_layer1 = nn.Sequential(nn.Linear(n_features, 100), nn.ReLU(), nn.Dropout(0.2))
        self.dense_layer2 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1).
        """
        # x is (B, C)
        x = self.dense_layer1(x)
        # -> x is (B, 100)
        out = self.dense_layer2(x)
        # -> out is (B, 1)

        return out
