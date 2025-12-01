"""
Models used by dense network classifiers.
"""

import torch
from torch import nn


class DenseSiteNet(nn.Module):
    """
    A dense neural network for site-based classification with improved regularization to prevent overfitting.

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
        input_dim = n_sites * n_features
        
        # First hidden layer with BatchNorm and increased dropout
        self.dense_layer1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Second hidden layer with BatchNorm and dropout
        self.dense_layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output layer (no dropout on output)
        self.dense_layer3 = nn.Linear(64, 1)

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
        # -> x is (B, 128)
        x = self.dense_layer2(x)
        # -> x is (B, 64)
        out = self.dense_layer3(x)
        # -> out is (B, 1)

        return out


class DenseMsaNet(nn.Module):
    """
    A dense neural network for MSA-based classification with improved regularization to prevent overfitting.

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
        
        # First hidden layer with BatchNorm and increased dropout
        self.dense_layer1 = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Second hidden layer with BatchNorm and dropout
        self.dense_layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output layer (no dropout on output)
        self.dense_layer3 = nn.Linear(64, 1)

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
        # -> x is (B, 128)
        x = self.dense_layer2(x)
        # -> x is (B, 64)
        out = self.dense_layer3(x)
        # -> out is (B, 1)

        return out
