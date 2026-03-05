import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------
# Residual Depthwise Separable Conv Block
# ---------------------------------------------------

class DSConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel,
            padding=kernel // 2,
            groups=in_ch,
            stride=stride
        )

        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)

        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x):

        residual = self.skip(x)

        x = self.depthwise(x)
        x = self.pointwise(x)

        x = self.norm(x)
        x = self.act(x)

        return x + residual


# ---------------------------------------------------
# Learnable Positional Encoding
# ---------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=5000):
        super().__init__()

        self.pos = nn.Parameter(
            torch.randn(1, max_len, dim) * 0.02
        )

    def forward(self, x):

        return x + self.pos[:, : x.size(1)]


# ---------------------------------------------------
# Attention Pooling
# ---------------------------------------------------

class GatedAttentionPooling(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.V = nn.Linear(dim, dim)
        self.U = nn.Linear(dim, dim)
        self.w = nn.Linear(dim, 1)

    def forward(self, x):

        A_V = torch.tanh(self.V(x))
        A_U = torch.sigmoid(self.U(x))

        A = self.w(A_V * A_U)

        weights = torch.softmax(A, dim=1)

        return torch.sum(weights * x, dim=1)


# ---------------------------------------------------
# CTENN Model
# ---------------------------------------------------

class CTENN_Murmur(nn.Module):

    def __init__(self):
        super().__init__()

        # -------------------------
        # CNN Frontend
        # -------------------------

        self.stem = nn.Conv1d(
            1, 32, kernel_size=21, stride=2, padding=10
        )

        self.blocks = nn.Sequential(

            DSConvBlock(32, 64),
            nn.MaxPool1d(2),

            DSConvBlock(64, 128),
            nn.MaxPool1d(2),

            DSConvBlock(128, 256),
            nn.MaxPool1d(2),
        )

        # -------------------------
        # Transformer
        # -------------------------

        hidden = 256

        self.positional = PositionalEncoding(hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=8,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )

        # -------------------------
        # Pooling
        # -------------------------

        self.pool = GatedAttentionPooling(hidden)

        # -------------------------
        # Classifier
        # -------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(hidden),

            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.GELU(),

            nn.Linear(64, 2)
        )

    def forward(self, wav):

        # (B,T) -> (B,1,T)
        x = wav.unsqueeze(1)

        x = self.stem(x)
        x = self.blocks(x)

        # (B,C,T) -> (B,T,C)
        x = x.transpose(1, 2)

        x = self.positional(x)

        x = self.transformer(x)

        x = self.pool(x)

        return self.classifier(x)