import torch
import torch.nn as nn
import torchaudio
from transformers import WhisperModel
from config import SAMPLE_RATE

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

        self.encoder = WhisperModel.from_pretrained(
            "openai/whisper-small"
        ).encoder

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        hidden = self.encoder.config.hidden_size

        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.post_lstm_norm = nn.LayerNorm(hidden)

        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
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

        mel_lin = self.mel(wav)
        frame_mask = (mel_lin.sum(dim=1) > 0)
        mel = torch.log(mel_lin + 1e-6)

        MAX_FRAMES = 3000
        T = mel.shape[-1]

        if T > MAX_FRAMES:
            mel = mel[..., :MAX_FRAMES]
            frame_mask = frame_mask[:, :MAX_FRAMES]
        else:
            pad_len = MAX_FRAMES - T
            mel = torch.nn.functional.pad(mel, (0, pad_len))
            frame_mask = torch.nn.functional.pad(
                frame_mask, (0, pad_len), value=False
            )

        feats = self.encoder(input_features=mel).last_hidden_state
        frame_mask = frame_mask[:, ::2][:, :feats.shape[1]]

        feats, _ = self.bilstm(feats)
        feats = self.post_lstm_norm(feats)

        feats = self.temporal(
            feats,
            src_key_padding_mask=~frame_mask
        )

        A_V = torch.tanh(self.attn_V(feats))
        A_U = torch.sigmoid(self.attn_U(feats))
        A = self.attn_w(A_V * A_U)

        A = A.masked_fill(~frame_mask.unsqueeze(-1), -1e9)

        weights = torch.softmax(A, dim=1)
        pooled = (weights * feats).sum(dim=1)

        return self.classifier(pooled)