import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from config import *
from data.dataset import MurmurDataset
from models.murmur_model import MurmurModel
from losses.focal_loss import FocalLoss
from metrics import validate_murmur_model

# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
dataset = MurmurDataset(AUDIO_DIR, META_JSON,
                        sr=SAMPLE_RATE, max_len=MAX_LEN)

labels = [dataset.meta[f] for f in dataset.files]

train_idx, val_idx = train_test_split(
    range(len(dataset)),
    test_size=0.1,
    random_state=42,
    stratify=labels
)

train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)

train_labels = [dataset.meta[dataset.files[i]] for i in train_idx]

class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts

sample_weights = [class_weights[l] for l in train_labels]

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples:   {len(val_ds)}")

# ---------------------------------------------------
# Training Function (UNCHANGED)
# ---------------------------------------------------
def train_murmur_model(model, train_loader, val_loader, epochs):
    model.train()

    labels = [y.item() for _, y in train_loader.dataset]
    pos = sum(labels)
    neg = len(labels) - pos

    weights = torch.tensor([1.0, neg / (pos + 1e-6)], device=DEVICE)
    alpha = weights / weights.sum()

    loss_fn = FocalLoss(alpha=alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    accum_steps = 4
    best_bal_acc = 0.0
    patience, wait = 3, 0

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for step, (wav, y) in enumerate(tqdm(train_loader, desc=f"Epoch {ep+1}")):
            wav, y = wav.to(DEVICE), y.to(DEVICE)

            logits = model(wav)
            loss = loss_fn(logits, y) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {ep+1} | Train Loss: {avg_train_loss:.4f}")

        val_loss, bal_acc = validate_murmur_model(
            model, val_loader, loss_fn, DEVICE
        )

        scheduler.step(val_loss)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            wait = 0
            torch.save(
                model.state_dict(),
                "/home/hariramanan/Desktop/madhan/hs_model/hs_murmur_bilstm_tp_gate_model_2khz_400_nfft.pt"
            )
            print("Best model saved")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break


# ---------------------------------------------------
# Run Training
# ---------------------------------------------------
model = MurmurModel().to(DEVICE)
train_murmur_model(model, train_dl, val_dl, epochs=20)