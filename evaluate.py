import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from config import *
from data.dataset import MurmurDataset
from models.murmur_model import MurmurModel
from metrics import evaluate_murmur_model

dataset = MurmurDataset(AUDIO_DIR, META_JSON,
                        sr=SAMPLE_RATE, max_len=MAX_LEN)

labels = [dataset.meta[f] for f in dataset.files]

train_idx, val_idx = train_test_split(
    range(len(dataset)),
    test_size=0.1,
    random_state=42,
    stratify=labels
)

val_ds = Subset(dataset, val_idx)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = MurmurModel().to(DEVICE)
model.load_state_dict(
    torch.load(
        "/home/hariramanan/Desktop/madhan/hs_model/hs_murmur_bilstm_tp_gate_model_2khz_400nfft.pt"
    )
)

evaluate_murmur_model(model, val_dl, DEVICE)