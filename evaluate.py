import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score
)

from config import *
from data.dataset import MurmurDataset
from qmodel.murmur_model import MurmurModel


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
dataset = MurmurDataset(
    AUDIO_DIR,
    META_JSON,
    sr=SAMPLE_RATE,
    window_sec=6,
    stride_sec=3
)

labels = [dataset.meta[f] for f in dataset.files]

_, val_idx = train_test_split(
    range(len(dataset)),
    test_size=0.1,
    random_state=42,
    stratify=labels
)

val_ds = Subset(dataset, val_idx)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)


# ---------------------------------------------------
# Generic Evaluation Function
# ---------------------------------------------------
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


# ---------------------------------------------------
# Evaluate FP32 Model
# ---------------------------------------------------
print("Evaluating FP32 model...")

fp32_model = MurmurModel().to(DEVICE)
fp32_model.load_state_dict(
    torch.load(
        "/home/hariramanan/Project/madhan/hs_model/hs_murmur_quartz_bilstm_tp_gate_model.pt",
        map_location=DEVICE
    )
)

fp32_results = evaluate_model(fp32_model, val_dl, DEVICE, use_fp16=False)


# ---------------------------------------------------
# Evaluate FP16 Model
# ---------------------------------------------------
print("Evaluating FP16 model...")

fp16_model = MurmurModel().to(DEVICE)

fp16_model.load_state_dict(
    torch.load(
        "/home/hariramanan/Project/madhan/hs_model/hs_murmur_fp16_quartz.pt",
        map_location=DEVICE
    )
)

fp16_model.eval()

# 🔥 Convert to half precision
fp16_model.half()

# 🔥 VERY IMPORTANT: keep MelSpectrogram in FP32
fp16_model.mel.float()
fp16_results = evaluate_model(fp16_model, val_dl, DEVICE, use_fp16=True)
# ---------------------------------------------------
# Generate Evaluation Report
# ---------------------------------------------------
report_path = "/home/hariramanan/Project/madhan/hs_model/evaluation_report_new.txt"

with open(report_path, "w") as f:

    f.write("====================================================\n")
    f.write(" Heart Sound Murmur Classification Evaluation Report\n")
    f.write("====================================================\n")
    f.write(f"Generated on: {datetime.now()}\n\n")

    f.write("----------------------------------------------------\n")
    f.write("FP32 MODEL RESULTS (GPU)\n")
    f.write("----------------------------------------------------\n")
    f.write(f"Balanced Accuracy: {fp32_results['balanced_accuracy']:.4f}\n")
    f.write(f"ROC-AUC: {fp32_results['roc_auc']:.4f}\n")
    f.write(f"F1 Score: {fp32_results['f1_score']:.4f}\n\n")
    f.write(fp32_results["classification_report"])
    f.write("\n\n")

    f.write("----------------------------------------------------\n")
    f.write("FP16 MODEL RESULTS (GPU)\n")
    f.write("----------------------------------------------------\n")
    f.write(f"Balanced Accuracy: {fp16_results['balanced_accuracy']:.4f}\n")
    f.write(f"ROC-AUC: {fp16_results['roc_auc']:.4f}\n")
    f.write(f"F1 Score: {fp16_results['f1_score']:.4f}\n\n")
    f.write(fp16_results["classification_report"])
    f.write("\n\n")

    f.write("----------------------------------------------------\n")
    f.write("PERFORMANCE DIFFERENCE (FP32 - FP16)\n")
    f.write("----------------------------------------------------\n")
    f.write(f"Balanced Accuracy Drop: "
            f"{fp32_results['balanced_accuracy'] - fp16_results['balanced_accuracy']:.4f}\n")
    f.write(f"ROC-AUC Drop: "
            f"{fp32_results['roc_auc'] - fp16_results['roc_auc']:.4f}\n")
    f.write(f"F1 Score Drop: "
            f"{fp32_results['f1_score'] - fp16_results['f1_score']:.4f}\n")

print(f"\nEvaluation report saved at:\n{report_path}")