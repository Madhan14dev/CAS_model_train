import os
import torch
import torch.quantization as tq
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
from models.murmur_model import MurmurModel


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
dataset = MurmurDataset(AUDIO_DIR, META_JSON,
                        sr=SAMPLE_RATE, max_len=MAX_LEN)

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
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for wav, y in loader:
            wav = wav.to(device)
            logits = model(wav)
            prob = torch.softmax(logits, dim=1)[:, 1]

            y_true.extend(y.numpy())
            y_prob.extend(prob.cpu().numpy())

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
# Load FP32 Model (GPU)
# ---------------------------------------------------
print("Evaluating FP32 model...")

fp32_model = MurmurModel().to(DEVICE)
fp32_model.load_state_dict(
    torch.load(
        "/home/hariramanan/Desktop/madhan/hs_model/hs_murmur_bilstm_tp_gate_model_2khz_400_nfft.pt",
        map_location=DEVICE
    )
)

fp32_results = evaluate_model(fp32_model, val_dl, DEVICE)


# ---------------------------------------------------
# Load Quantized INT8 Model (CPU)
# ---------------------------------------------------
print("Evaluating INT8 Quantized model...")

int8_model = MurmurModel()

# int8_model = tq.quantize_dynamic(
#     int8_model,
#     {torch.nn.Linear, torch.nn.LSTM},
#     dtype=torch.qint8
# )
int8_model.bilstm = tq.quantize_dynamic(
    int8_model.bilstm,
    {torch.nn.LSTM},
    dtype=torch.qint8
)

int8_model.classifier = tq.quantize_dynamic(
    int8_model.classifier,
    {torch.nn.Linear},
    dtype=torch.qint8
)

int8_model.load_state_dict(
        torch.load(
            "/home/hariramanan/Desktop/madhan/hs_model/hs_murmur_quantized_int8.pt",
            map_location="cpu",
            weights_only=False
        )
)

int8_results = evaluate_model(int8_model, val_dl, device="cpu")


# ---------------------------------------------------
# Generate Evaluation Report
# ---------------------------------------------------
report_path = "/home/hariramanan/Desktop/madhan/hs_model/evaluation_report.txt"

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
    f.write("INT8 QUANTIZED MODEL RESULTS (CPU)\n")
    f.write("----------------------------------------------------\n")
    f.write(f"Balanced Accuracy: {int8_results['balanced_accuracy']:.4f}\n")
    f.write(f"ROC-AUC: {int8_results['roc_auc']:.4f}\n")
    f.write(f"F1 Score: {int8_results['f1_score']:.4f}\n\n")
    f.write(int8_results["classification_report"])
    f.write("\n\n")

    f.write("----------------------------------------------------\n")
    f.write("PERFORMANCE DIFFERENCE\n")
    f.write("----------------------------------------------------\n")
    f.write(f"Balanced Accuracy Drop: "
            f"{fp32_results['balanced_accuracy'] - int8_results['balanced_accuracy']:.4f}\n")
    f.write(f"ROC-AUC Drop: "
            f"{fp32_results['roc_auc'] - int8_results['roc_auc']:.4f}\n")
    f.write(f"F1 Score Drop: "
            f"{fp32_results['f1_score'] - int8_results['f1_score']:.4f}\n")

print(f"\nEvaluation report saved at:\n{report_path}")