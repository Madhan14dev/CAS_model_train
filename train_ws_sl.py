import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score
)
from config import *
from data.dataset import MurmurDataset
from models.murmur_model_copy import MurmurModel
from losses.focal_loss import FocalLoss
from metrics import validate_murmur_model
from pathlib import Path


# BASE_DIR = Path("/home/hariramanan/Project/madhan")
CHECKPOINT_PATH = "/home/hariramanan/Project/madhan/hs_model/murmur_checkpoint_small_new.pt"
BEST_MODEL_PATH = "/home/hariramanan/Project/madhan/hs_model/hs_murmur_wb_sw_bilstm_tp_gate_model_small_new.pt"
# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
dataset = MurmurDataset(
    AUDIO_DIR,
    META_JSON,
    sr=SAMPLE_RATE,
    window_sec=window_sec,
    stride_sec=stride_sec
)

# Extract labels for each segment
labels = [label for _, label in dataset.samples]
indices = np.arange(len(dataset))

# ---- First split: Train vs Temp (Val + Test) ----
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.1,          # 10% reserved for val + test
    random_state=42,
    stratify=labels
)

# Labels for temp split
temp_labels = [labels[i] for i in temp_idx]

# ---- Second split: Val vs Test ----
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,          # 10% val, 10% test
    random_state=42,
    stratify=temp_labels
)

# Create dataset subsets
train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)
test_ds  = Subset(dataset, test_idx)

# Training labels
train_labels = [labels[i] for i in train_idx]

# ---- Class balancing for training ----
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts

sample_weights = [class_weights[l] for l in train_labels]

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ---- DataLoaders ----
train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples:   {len(val_ds)}")
print(f"Test samples:  {len(test_ds)}")

def evaluate_model(model, loader, device, use_fp16=False):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for wav, y in loader:
            wav = wav.to(device)

            if use_fp16:
                wav = wav.half()

            logits = model(wav)
            prob = torch.softmax(logits, dim=1)[:, 1]

            y_true.extend(y.numpy())
            y_prob.extend(prob.detach().cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_pred = (y_prob > 0.5).astype(int)

    results = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1_score": f1_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=4)
    }

    return results

def save_checkpoint(epoch, model, optimizer, scheduler, best_bal_acc):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_bal_acc": best_bal_acc
    }

    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved at epoch {epoch+1}")
def load_checkpoint(model, optimizer, scheduler):
    if os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint...")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        start_epoch = checkpoint["epoch"] + 1
        best_bal_acc = checkpoint["best_bal_acc"]

        print(f"Resuming training from epoch {start_epoch}")

        return start_epoch, best_bal_acc

    return 0, 0.0
# ---------------------------------------------------
# Training Function (UNCHANGED)
# ---------------------------------------------------
def train_murmur_model(model, train_loader, val_loader, epochs):

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
    patience = 3
    wait = 0

    # 🔹 Load checkpoint if exists
    start_epoch, best_bal_acc = load_checkpoint(model, optimizer, scheduler)

    for ep in range(start_epoch, epochs):

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

        # 🔹 Save checkpoint every epoch
        save_checkpoint(ep, model, optimizer, scheduler, best_bal_acc)

        # 🔹 Save best model
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            wait = 0

            torch.save(model.state_dict(), BEST_MODEL_PATH)
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
results = evaluate_model(model, test_dl, DEVICE, use_fp16=False)
# ---------------------------------------------------
# Generate Evaluation Report
# ---------------------------------------------------
report_path = "/home/hariramanan/Project/madhan/hs_model/evaluation_report_whisper_small_new.txt"

with open(report_path, "w") as f:

    f.write("====================================================\n")
    f.write(" Heart Sound Murmur Classification Evaluation Report\n")
    f.write("====================================================\n")
    f.write(f"Generated on: {datetime.now()}\n\n")

    f.write("----------------------------------------------------\n")
    f.write("FP32 MODEL RESULTS (GPU)\n")
    f.write("----------------------------------------------------\n")
    f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
    f.write(f"ROC-AUC: {results['roc_auc']:.4f}\n")
    f.write(f"F1 Score: {results['f1_score']:.4f}\n\n")
    f.write(results["classification_report"])
    f.write("\n\n")
print(f"Evaluation report saved at:\n{report_path}")    