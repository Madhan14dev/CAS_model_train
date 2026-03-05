# models/murmur_model.py

import torch
import torch.nn as nn
import torchaudio

from config import SAMPLE_RATE
from .quartznet_encoder import QuartzNetEncoder


class MurmurModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,
            hop_length=80,
            n_mels=80,
            f_min=20,
            f_max=800
        )

        # QuartzNet Encoder
        self.encoder = QuartzNetEncoder(repeat=2)

        hidden = 1024

        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.post_lstm_norm = nn.LayerNorm(hidden)

        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=8,
                dim_feedforward=hidden * 2,
                batch_first=True,
                activation="gelu",
                dropout=0.1
            ),
            num_layers=2
        )

        self.attn_V = nn.Linear(hidden, hidden)
        self.attn_U = nn.Linear(hidden, hidden)
        self.attn_w = nn.Linear(hidden, 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, wav):

        mel = self.mel(wav.float())
        mel = torch.log(mel + 1e-6)

        # QuartzNet expects (B, n_mels, T)
        feats = self.encoder(mel)

        feats, _ = self.bilstm(feats)
        feats = self.post_lstm_norm(feats)

        feats = self.temporal(feats)

        A_V = torch.tanh(self.attn_V(feats))
        A_U = torch.sigmoid(self.attn_U(feats))
        A = self.attn_w(A_V * A_U)

        weights = torch.softmax(A, dim=1)
        pooled = (weights * feats).sum(dim=1)

        return self.classifier(pooled)