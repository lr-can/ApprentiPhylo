"""
Models for convolution network classifiers.
"""

import torch
from torch import nn


class AAConvNet(nn.Module):
    """
    A convolutional neural network for amino acid sequence classification.

    This class implements a simple convolutional neural network with one convolutional layer
    followed by a dense layer for binary classification of amino acid sequences.

    Parameters
    ----------
    n_features : int
        The number of input features (channels) for each amino acid position.
    n_sites : int
        The number of amino acid positions (sequence length) in the input.
    kernel_size : int, default=1
        The size of the convolutional kernel.
    """

    def __init__(
        self,
        n_features: int,
        n_sites: int,
        kernel_size: int = 1,
    ):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.2),
        )

        self.dense_layer = nn.Linear(64, 1)

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
            Output tensor of shape (batch_size, 1) containing the classification scores.
        """
        # x is (B, S, C)
        x = x.swapaxes(1, 2)
        # -> x is (B, C, S)
        x = self.conv_layer(x)
        # -> x is (B, 64, 1)
        x = x.reshape(x.size(0), -1)
        # -> x is (B, 64)
        out = self.dense_layer(x)
        # -> out is (B, 1)

        return out


class DNAConvNet(nn.Module):
    """
    A convolutional neural network for DNA sequence classification.

    This class implements a convolutional neural network with two convolutional layers
    followed by a dense layer for binary classification of DNA sequences.

    Parameters
    ----------
    n_features : int
        The number of input features (channels) for each DNA position.
    n_sites : int
        The number of DNA positions (sequence length) in the input.
    """

    def __init__(
        self,
        n_features: int,
        n_sites: int,
    ):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(n_features, 100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(100, 210, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=n_sites),
            nn.Dropout(0.2),
        )

        self.dense_layer = nn.Linear(210, 1)

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
            Output tensor of shape (batch_size, 1) containing the classification scores.
        """
        # x is (B, S, C)
        x = x.swapaxes(1, 2)
        # -> x is (B, C, S)
        x = self.conv_layer1(x)
        # -> x is (B, 100, S)
        x = self.conv_layer2(x)
        # -> x is (B, 210, 1)
        x = x.reshape(x.size(0), -1)
        # -> x is (B, 210)
        out = self.dense_layer(x)
        # -> out is (B, 1)

        return out
