"""
Model used by attention based classifiers. Architecture taken from Phyloformer.
"""

import torch
from torch import nn


class FixedKernelMultiAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.nb_heads = nb_heads
        self.dropout = dropout
        self.head_dim = embed_dim // nb_heads

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.elu = nn.ELU()
        self.eps = eps

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.atten_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, input):
        batch_size, nb_row, nb_col, embed_dim = input.size()

        k = self.k_proj(input).view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim).transpose(2, 3)
        q = self.q_proj(input).view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim).transpose(2, 3)
        v = self.v_proj(input).view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim).transpose(2, 3)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        KtV = k.transpose(-1, -2) @ v

        Z = 1 / (q @ k.transpose(-1, -2).sum(dim=-1, keepdim=True) + self.eps)
        Z = Z.expand(batch_size, nb_row, self.nb_heads, nb_col, self.head_dim)

        # FIX
        V = Z * (q @ KtV)
        V = V.transpose(2, 3).contiguous().view(batch_size, -1, nb_col, embed_dim)

        out = self.proj_drop(self.out_proj(V))

        return out


class AttentionNet(nn.Module):
    def __init__(
        self,
        *,
        dropout=0.0,
        n_blocks=4,
        h_dim=32,
        n_heads=4,
        n_classes: int = 2,
        n_features: int,
        device="cpu",
        disable_row_attention=False,
        disable_column_attention=False,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.n_features = n_features
        self.rowAttentions = nn.ModuleList()
        self.columnAttentions = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.fNNs = nn.ModuleList()
        self.disable_row_attention = disable_row_attention
        self.disable_column_attention = disable_column_attention

        layers_1_1 = [
            nn.Conv2d(in_channels=self.n_features, out_channels=h_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        ]
        self.block_1_1 = nn.Sequential(*layers_1_1)
        self.norm = nn.LayerNorm(h_dim)
        self.pwFNN = nn.Sequential(
            *[
                nn.Conv2d(in_channels=h_dim, out_channels=self.n_classes, kernel_size=1, stride=1),
            ]
        )
        for _ in range(self.n_blocks):
            self.rowAttentions.append(FixedKernelMultiAttention(h_dim, n_heads).to(device))
            self.columnAttentions.append(FixedKernelMultiAttention(h_dim, n_heads).to(device))
            self.layernorms.append(nn.LayerNorm(h_dim).to(device))
            self.fNNs.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            in_channels=h_dim, out_channels=h_dim * 4, kernel_size=1, stride=1, device=device
                        ),
                        nn.Dropout(dropout),
                        nn.GELU(),
                        nn.Conv2d(
                            in_channels=h_dim * 4, out_channels=h_dim, kernel_size=1, stride=1, device=device
                        ),
                    ],
                    nn.Dropout(dropout),
                )
            )

        self.device = device

    def forward(self, x):
        # Convert to one-hot encoding and reshape
        x = torch.nn.functional.one_hot(x, num_classes=self.n_features)
        x = torch.moveaxis(x, 3, 1)
        x = x.float()

        # input tensor has shape (batch_size,n_features,n_sites,n_seq)
        out = self.block_1_1(
            x
        )  # 2d convolution that sgives us the features in the third dimension (i.e. initial embedding
        # of each amino acid) from here on the tensor has shape (batch_size,h_dim,n_sites,n_seq),
        # all the transpose/permute allow to apply layernorm and attention over the desired dimensions
        # and are then followed by the inverse transposition/permutation of dimensions

        out = self.norm(out.transpose(-1, -3)).transpose(-1, -3)  # layernorm

        for i in range(self.n_blocks):
            # AXIAL ATTENTIONS BLOCK
            # ----------------------
            # ROW ATTENTION
            if not self.disable_row_attention:
                att = self.rowAttentions[i](out.permute(0, 2, 3, 1))
                out = att.permute(0, 3, 1, 2) + out  # row attention+residual connection
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(-1, -3)  # layernorm

            # COLUMN ATTENTION
            if not self.disable_column_attention:
                att = self.columnAttentions[i](out.permute(0, 3, 2, 1))
                out = att.permute(0, 3, 2, 1) + out  # column attention+residual connection
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(-1, -3)  # layernorm

            # FEEDFORWARD
            out = self.fNNs[i](out) + out
            if i != self.n_blocks - 1:
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(-1, -3)  # layernorm

        out = self.pwFNN(out)  # after this last convolution we have (batch_size,n_classes,n_sites,n_seq)
        out = torch.squeeze(torch.mean(out, dim=-1))  # averaging over sequences and removing the extra
        # dimensions we finally get (batch_size,n_classes,n_sites)
        out = torch.squeeze(out[:, :, -1])  # taking last token embedding as sequence embedding
        # (batch_size,n_classes)
        out = torch.squeeze(out[:, -1])  # taking class 1 logit only
        # (batch_size)
        return out
