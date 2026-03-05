import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    brier_score_loss,
    precision_score
)

# ---------------------------------------------------
# Threshold Search
# ---------------------------------------------------
def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 100):
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ---------------------------------------------------
# Validation
# ---------------------------------------------------
def validate_murmur_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []

    with torch.no_grad():
        for wav, y in loader:
            wav, y = wav.to(device), y.to(device)
            logits = model(wav)
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            prob = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(y.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    best_t = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob > best_t).astype(int)

    avg_loss = total_loss / len(loader)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print(
        f"Val Loss: {avg_loss:.4f} | "
        f"Balanced Acc: {bal_acc:.4f} | "
        f"Best T: {best_t:.2f}"
    )

    return avg_loss, bal_acc


# ---------------------------------------------------
# Sensitivity @ Specificity
# ---------------------------------------------------
def sensitivity_at_specificity(y_true, y_prob, target_spec=0.85):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr

    idx = np.where(specificity >= target_spec)[0]

    if len(idx) == 0:
        return None, None

    best_idx = idx[-1]
    return tpr[best_idx], thresholds[best_idx]


# ---------------------------------------------------
# Calibration Plot
# ---------------------------------------------------
def plot_calibration(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    brier = brier_score_loss(y_true, y_prob)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration Curve (Brier={brier:.4f})")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Brier score:", round(brier, 4))


# ---------------------------------------------------
# Full Evaluation
# ---------------------------------------------------
def evaluate_murmur_model(model, loader, device, target_spec=0.85):
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

    best_f1_t = find_best_threshold(y_true, y_prob)
    y_pred_f1 = (y_prob > best_f1_t).astype(int)

    print("\n===== F1-OPTIMAL THRESHOLD EVALUATION =====")
    print(f"Optimal threshold (F1): {best_f1_t:.3f}")
    print(classification_report(y_true, y_pred_f1, digits=4))
    print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred_f1))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

    print("\n===== CALIBRATION =====")
    plot_calibration(y_true, y_prob)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr

    valid = np.where(specificity >= target_spec)[0]

    if len(valid) == 0:
        print(f"\nSensitivity @ {int(target_spec*100)}% specificity: NOT ACHIEVABLE")
        return

    idx = valid[-1]
    clinical_t = thresholds[idx]
    sens = tpr[idx]

    y_pred_clinical = (y_prob > clinical_t).astype(int)

    print("\n===== CLINICAL OPERATING POINT =====")
    print(f"Target specificity: {int(target_spec*100)}%")
    print(f"Operating threshold: {clinical_t:.3f}")
    print(f"Sensitivity @ {int(target_spec*100)}% spec: {sens:.3f}")
    print(
        "Precision @ clinical point:",
        precision_score(y_true, y_pred_clinical)
    )

    cm = confusion_matrix(y_true, y_pred_clinical)

    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix @ {int(target_spec*100)}% Specificity")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.show()