from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# =========================
# Project paths
# =========================
PROJECT_ROOT = Path(r"C:\Users\hebas\OneDrive\Desktop\Project ECG")

DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models"
REPORTS_DIR = PROJECT_ROOT / "Reports"
OUT_DIR = REPORTS_DIR / "chapter12_figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Transformer custom layer
# Needed only if Transformer model exists
# =========================
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model

        pos = np.arange(length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(
            10000,
            (2 * (i // 2)) / np.float32(d_model)
        )
        angles = pos * angle_rates

        pe = np.zeros((length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :])

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "length": self.length,
            "d_model": self.d_model,
        })
        return cfg


# =========================
# Data helpers
# =========================
def load_ptbdb_test_split():
    normal_path = DATASET_DIR / "ptbdb_normal.csv"
    abnormal_path = DATASET_DIR / "ptbdb_abnormal.csv"

    if not normal_path.exists():
        raise FileNotFoundError(f"Missing file: {normal_path}")

    if not abnormal_path.exists():
        raise FileNotFoundError(f"Missing file: {abnormal_path}")

    df_n = pd.read_csv(normal_path, header=None)
    df_a = pd.read_csv(abnormal_path, header=None)

    # First 187 columns are ECG beat features
    x_n = df_n.iloc[:, :187].to_numpy(dtype=np.float32)
    x_a = df_a.iloc[:, :187].to_numpy(dtype=np.float32)

    # Labels:
    # Normal = 0
    # Abnormal = 1
    y_n = np.zeros(len(x_n), dtype=np.int64)
    y_a = np.ones(len(x_a), dtype=np.int64)

    x = np.vstack([x_n, x_a])
    y = np.concatenate([y_n, y_a])

    # Same evaluation idea as training scripts:
    # use stratified test split so normal/abnormal distribution is preserved.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return x_test, y_test


def normalize_beats(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x, axis=1, keepdims=True)
    x = x - med
    scale = np.max(np.abs(x), axis=1, keepdims=True) + 1e-9
    x = x / scale
    return x.astype(np.float32)


def predict_binary_model(model, x):
    x3 = np.asarray(x, dtype=np.float32).reshape(-1, 187, 1)
    probs = model.predict(x3, batch_size=256, verbose=0).reshape(-1)
    return probs.astype(np.float32)


# =========================
# Final live pipeline prediction
# Same logic as app:
# TCN first.
# If confident -> use TCN.
# If uncertain -> TCN + ResNet + Transformer weighted average.
# =========================
def predict_final_live_pipeline(x_test):
    resnet_path = MODELS_DIR / "ptbdb_resnet1d_fix.keras"
    tcn_path = MODELS_DIR / "ptbdb_tcn.keras"
    transformer_path = MODELS_DIR / "ptbdb_transformer_fix.keras"

    if not resnet_path.exists():
        raise FileNotFoundError(f"Missing required model: {resnet_path}")

    print("Loading ResNet1D...")
    resnet = tf.keras.models.load_model(resnet_path, compile=False)

    tcn = None
    transformer = None

    if tcn_path.exists():
        print("Loading TCN...")
        tcn = tf.keras.models.load_model(tcn_path, compile=False)
    else:
        print("TCN not found. The script will use ResNet only.")

    if transformer_path.exists():
        print("Loading Transformer...")
        transformer = tf.keras.models.load_model(
            transformer_path,
            compile=False,
            custom_objects={"PositionalEncoding": PositionalEncoding},
        )
    else:
        print("Transformer not found. Ensemble will use available models only.")

    x_test = normalize_beats(x_test)

    # If TCN is missing, use ResNet directly
    if tcn is None:
        return predict_binary_model(resnet, x_test)

    p_tcn = predict_binary_model(tcn, x_test)

    # Uncertainty range used by the app
    unc_low = 0.20
    unc_high = 0.85

    p_final = np.zeros_like(p_tcn, dtype=np.float32)

    confident_mask = (p_tcn < unc_low) | (p_tcn > unc_high)
    uncertain_mask = ~confident_mask

    # Confident TCN predictions
    p_final[confident_mask] = p_tcn[confident_mask]

    # Uncertain predictions: weighted ensemble
    if np.any(uncertain_mask):
        x_unc = x_test[uncertain_mask]

        weighted_probs = []
        weighted_probs.append((0.50, p_tcn[uncertain_mask]))

        p_res = predict_binary_model(resnet, x_unc)
        weighted_probs.append((0.30, p_res))

        if transformer is not None:
            p_tr = predict_binary_model(transformer, x_unc)
            weighted_probs.append((0.20, p_tr))

        total_w = sum(w for w, _ in weighted_probs)
        p_unc = sum(w * p for w, p in weighted_probs) / max(total_w, 1e-9)

        p_final[uncertain_mask] = p_unc.astype(np.float32)

    return p_final


# =========================
# Figure helpers
# =========================
def plot_confusion_matrix(cm, labels, title, out_path):
    cm = np.asarray(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)


def save_text(text, out_path):
    out_path.write_text(text, encoding="utf-8")
    print("Saved:", out_path)


# =========================
# Main evaluation
# =========================
def main():
    print("Loading PTBDB test split...")
    x_test, y_true = load_ptbdb_test_split()

    print("Predicting using final live pipeline...")
    probs = predict_final_live_pipeline(x_test)

    threshold = 0.60
    y_pred = (probs >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, probs)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        digits=4,
        zero_division=0,
    )

    # Save confusion matrix figure
    plot_confusion_matrix(
        cm,
        labels=["Normal", "Abnormal"],
        title="PTBDB Final Live Pipeline Confusion Matrix",
        out_path=OUT_DIR / "Figure_12_1_PTBDB_Final_Live_Pipeline_Confusion_Matrix.png",
    )

    # Save summary text
    summary_text = (
        "PTBDB Final Live Pipeline Test Result\n"
        f"Samples: {len(y_true)}\n"
        f"Threshold: {threshold:.2f}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Precision abnormal: {precision:.4f}\n"
        f"Recall abnormal: {recall:.4f}\n"
        f"F1 abnormal: {f1:.4f}\n"
        f"ROC-AUC: {roc_auc:.4f}\n"
        f"Confusion Matrix: {cm.tolist()}\n\n"
        "Classification Report:\n"
        f"{report}\n"
    )

    save_text(
        summary_text,
        OUT_DIR / "PTBDB_Final_Live_Pipeline_Summary.txt",
    )

    # Save JSON
    result_json = {
        "model": "PTBDB Final Live Pipeline",
        "threshold": threshold,
        "samples": int(len(y_true)),
        "accuracy": float(acc),
        "precision_abnormal": float(precision),
        "recall_abnormal": float(recall),
        "f1_abnormal": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.astype(int).tolist(),
    }

    json_path = OUT_DIR / "PTBDB_Final_Live_Pipeline_Result.json"
    json_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    print("Saved:", json_path)

    print("\nDone. Check this folder:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()