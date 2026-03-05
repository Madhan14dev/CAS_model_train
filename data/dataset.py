import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from config import SAMPLE_RATE,window_sec,stride_sec

class MurmurDataset(Dataset):
    def __init__(self, audio_dir, meta_json,
                 sr=SAMPLE_RATE,
                 window_sec=window_sec,
                 stride_sec=stride_sec):

        with open(meta_json, "r") as f:
            self.meta = json.load(f)

        self.audio_dir = audio_dir
        self.files = list(self.meta.keys())

        self.sr = sr
        self.window_size = int(window_sec * sr)
        self.stride = int(stride_sec * sr)

        self.samples = []

        for fname in self.files:

            path = os.path.join(self.audio_dir, fname)
            wav, orig_sr = torchaudio.load(path)
            wav = wav.mean(0)

            # Resample
            if orig_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_sr, sr)

            # Normalize
            wav = wav / (wav.abs().max() + 1e-6)

            label = self.meta[fname]
            L = wav.shape[0]

            if L < self.window_size:

                repeat = self.window_size // L + 1
                wav = wav.repeat(repeat)[:self.window_size]

                self.samples.append((wav, label))

            else:

                starts = list(range(0, L - self.window_size + 1, self.stride))

                for start in starts:
                    segment = wav[start:start+self.window_size]
                    self.samples.append((segment, label))

                if starts[-1] + self.window_size < L:

                    tail_start = L - self.window_size
                    segment = wav[tail_start:tail_start + self.window_size]

                    self.samples.append((segment, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav, label = self.samples[idx]
        return wav, torch.tensor(label, dtype=torch.long)