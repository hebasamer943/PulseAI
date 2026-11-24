from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# =========================
# Transformer custom layer
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
            10000, (2 * (i // 2)) / np.float32(d_model)
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
        cfg.update({"length": self.length, "d_model": self.d_model})
        return cfg


# =========================
# Helpers
# =========================

def ensure_187(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size > 187:
        x = x[:187]
    if x.size < 187:
        x = np.pad(x, (0, 187 - x.size))
    return x.astype(np.float32)


def z_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    best = None
    for threshold in np.arange(0.05, 0.96, 0.01):
        result = binary_metrics(y_true, y_prob, float(threshold))
        if best is None:
            best = result
            continue

        # نختار أفضل F1، وإذا تعادلوا نأخذ الأعلى accuracy
        if (
            result["f1_score"] > best["f1_score"]
            or (
                result["f1_score"] == best["f1_score"]
                and result["accuracy"] > best["accuracy"]
            )
        ):
            best = result

    return best


def predict_binary_model(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1, 187, 1)
    prob = model.predict(x, verbose=0, batch_size=batch_size).reshape(-1)
    return prob.astype(np.float32)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# =========================
# Load datasets
# =========================

def load_ptbdb(dataset_dir: Path):
    normal_path = dataset_dir / "ptbdb_normal.csv"
    abnormal_path = dataset_dir / "ptbdb_abnormal.csv"

    if not normal_path.exists():
        raise FileNotFoundError(f"Missing: {normal_path}")
    if not abnormal_path.exists():
        raise FileNotFoundError(f"Missing: {abnormal_path}")

    df_n = pd.read_csv(normal_path, header=None)
    df_a = pd.read_csv(abnormal_path, header=None)

    x_n = df_n.iloc[:, :187].to_numpy(dtype=np.float32)
    x_a = df_a.iloc[:, :187].to_numpy(dtype=np.float32)

    y_n = np.zeros(len(x_n), dtype=np.int64)
    y_a = np.ones(len(x_a), dtype=np.int64)

    x = np.vstack([x_n, x_a]).astype(np.float32)
    y = np.concatenate([y_n, y_a]).astype(np.int64)

    return x, y


def load_mitbih_test(dataset_dir: Path):
    test_path = dataset_dir / "mitbih_test.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing: {test_path}")

    df = pd.read_csv(test_path, header=None)

    x = df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, 187].to_numpy(dtype=np.int64)

    return x, y_raw


# =========================
# PTBDB models
# =========================

def load_ptbdb_models(models_dir: Path):
    models = {}

    resnet_path = models_dir / "ptbdb_resnet1d_fix.keras"
    tcn_path = models_dir / "ptbdb_tcn.keras"
    transformer_path = models_dir / "ptbdb_transformer_fix.keras"

    if resnet_path.exists():
        models["ResNet1D"] = tf.keras.models.load_model(resnet_path, compile=False)
        print(f"Loaded: {resnet_path.name}")
    else:
        print(f"Missing: {resnet_path.name}")

    if tcn_path.exists():
        models["TCN"] = tf.keras.models.load_model(tcn_path, compile=False)
        print(f"Loaded: {tcn_path.name}")
    else:
        print(f"Missing: {tcn_path.name}")

    if transformer_path.exists():
        models["Transformer"] = tf.keras.models.load_model(
            transformer_path,
            compile=False,
            custom_objects={"PositionalEncoding": PositionalEncoding},
        )
        print(f"Loaded: {transformer_path.name}")
    else:
        print(f"Missing: {transformer_path.name}")

    return models


def predict_tcn_first_ensemble(models: dict, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """
    Matches the current application logic:
    1. TCN first.
    2. If TCN is confident, use it directly.
    3. If TCN is uncertain, combine TCN + ResNet + Transformer.

    Uncertainty range: 0.20 to 0.85
    Weights:
        TCN = 0.50
        ResNet1D = 0.30
        Transformer = 0.20
    """

    if "TCN" not in models and "ResNet1D" not in models:
        raise RuntimeError("Need at least TCN or ResNet1D model for PTBDB evaluation.")

    if "TCN" not in models:
        return predict_binary_model(models["ResNet1D"], x, batch_size=batch_size)

    p_tcn = predict_binary_model(models["TCN"], x, batch_size=batch_size)
    p_final = p_tcn.copy()

    uncertain = (p_tcn >= 0.20) & (p_tcn <= 0.85)

    if np.any(uncertain):
        weighted = [(0.50, p_tcn[uncertain])]

        if "ResNet1D" in models:
            p_res = predict_binary_model(models["ResNet1D"], x[uncertain], batch_size=batch_size)
            weighted.append((0.30, p_res))

        if "Transformer" in models:
            p_tr = predict_binary_model(models["Transformer"], x[uncertain], batch_size=batch_size)
            weighted.append((0.20, p_tr))

        total_w = sum(w for w, _ in weighted)
        p_mix = sum(w * p for w, p in weighted) / max(total_w, 1e-9)
        p_final[uncertain] = p_mix

    return p_final.astype(np.float32)


def evaluate_ptbdb(project_root: Path, batch_size: int = 256):
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"

    x, y = load_ptbdb(dataset_dir)

    # نفس فكرة training scripts: held-out 20% test split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = load_ptbdb_models(models_dir)

    rows = []
    confusion_rows = []

    for name, model in models.items():
        prob = predict_binary_model(model, x_test, batch_size=batch_size)

        fixed = binary_metrics(y_test, prob, threshold=0.60)
        best = find_best_threshold(y_test, prob)

        rows.append({
            "dataset": "PTBDB",
            "task": "Normal vs Abnormal",
            "model": name,
            "threshold_used": fixed["threshold"],
            "best_threshold": best["threshold"],
            "samples": fixed["samples"],
            "accuracy": fixed["accuracy"],
            "precision": fixed["precision"],
            "recall": fixed["recall"],
            "specificity": fixed["specificity"],
            "f1_score": fixed["f1_score"],
            "roc_auc": fixed["roc_auc"],
        })

        confusion_rows.append({
            "dataset": "PTBDB",
            "model": name,
            "threshold": fixed["threshold"],
            "tn": fixed["tn"],
            "fp": fixed["fp"],
            "fn": fixed["fn"],
            "tp": fixed["tp"],
        })

    # Final live pipeline
    ensemble_prob = predict_tcn_first_ensemble(models, x_test, batch_size=batch_size)
    fixed = binary_metrics(y_test, ensemble_prob, threshold=0.60)
    best = find_best_threshold(y_test, ensemble_prob)

    rows.append({
        "dataset": "PTBDB",
        "task": "Normal vs Abnormal",
        "model": "Final Live Pipeline: TCN-first Conditional Ensemble",
        "threshold_used": fixed["threshold"],
        "best_threshold": best["threshold"],
        "samples": fixed["samples"],
        "accuracy": fixed["accuracy"],
        "precision": fixed["precision"],
        "recall": fixed["recall"],
        "specificity": fixed["specificity"],
        "f1_score": fixed["f1_score"],
        "roc_auc": fixed["roc_auc"],
    })

    confusion_rows.append({
        "dataset": "PTBDB",
        "model": "Final Live Pipeline: TCN-first Conditional Ensemble",
        "threshold": fixed["threshold"],
        "tn": fixed["tn"],
        "fp": fixed["fp"],
        "fn": fixed["fn"],
        "tp": fixed["tp"],
    })

    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


# =========================
# MIT-BIH Stage 1
# =========================

def evaluate_mitbih_stage1(project_root: Path, batch_size: int = 256):
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"

    model_path = models_dir / "mitbih_stage1_binary_v2.keras"
    meta_path = models_dir / "mitbih_stage1_binary_v2_meta.json"

    if not model_path.exists():
        print("Skipping MIT-BIH Stage 1: model missing.")
        return pd.DataFrame(), pd.DataFrame()

    x_test, y_raw = load_mitbih_test(dataset_dir)

    # MIT-BIH binary:
    # 0 = Normal
    # 1,2,3,4 = Abnormal
    y_test = (y_raw != 0).astype(np.int64)

    x_test = z_norm(x_test).reshape(-1, 187, 1)

    meta = load_json(meta_path)
    threshold = float(meta.get("best_threshold_from_val", 0.76))

    model = tf.keras.models.load_model(model_path, compile=False)
    prob = model.predict(x_test, verbose=0, batch_size=batch_size).reshape(-1)

    fixed = binary_metrics(y_test, prob, threshold=threshold)
    best = find_best_threshold(y_test, prob)

    rows = [{
        "dataset": "MIT-BIH",
        "task": "Stage 1: Normal-like vs Abnormal",
        "model": "MIT-BIH Stage 1 Binary v2",
        "threshold_used": fixed["threshold"],
        "best_threshold": best["threshold"],
        "samples": fixed["samples"],
        "accuracy": fixed["accuracy"],
        "precision": fixed["precision"],
        "recall": fixed["recall"],
        "specificity": fixed["specificity"],
        "f1_score": fixed["f1_score"],
        "roc_auc": fixed["roc_auc"],
    }]

    confusion_rows = [{
        "dataset": "MIT-BIH",
        "model": "MIT-BIH Stage 1 Binary v2",
        "threshold": fixed["threshold"],
        "tn": fixed["tn"],
        "fp": fixed["fp"],
        "fn": fixed["fn"],
        "tp": fixed["tp"],
    }]

    return pd.DataFrame(rows), pd.DataFrame(confusion_rows)


# =========================
# MIT-BIH Stage 2
# =========================

def evaluate_mitbih_stage2(project_root: Path, batch_size: int = 256):
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"

    model_path = models_dir / "mitbih_stage2_subtypes_svf_v2.keras"
    meta_path = models_dir / "mitbih_stage2_subtypes_svf_v2_meta.json"

    if not model_path.exists():
        print("Skipping MIT-BIH Stage 2: model missing.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x_test, y_raw = load_mitbih_test(dataset_dir)

    # Keep only S, V, F:
    # 1 = S, 2 = V, 3 = F
    mask = np.isin(y_raw, [1, 2, 3])
    x_test = x_test[mask]
    y_raw = y_raw[mask]

    label_map = {1: 0, 2: 1, 3: 2}
    label_names = ["S", "V", "F"]
    display_names = {
        "S": "Supraventricular ectopic",
        "V": "Ventricular ectopic",
        "F": "Fusion beat",
    }

    y_test = np.asarray([label_map[int(v)] for v in y_raw], dtype=np.int64)

    x_test = z_norm(x_test).reshape(-1, 187, 1)

    meta = load_json(meta_path)
    conf_threshold = float(meta.get("recommended_confidence_threshold", 0.90))

    model = tf.keras.models.load_model(model_path, compile=False)
    probs = model.predict(x_test, verbose=0, batch_size=batch_size)

    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    known_accuracy = accuracy_score(y_test, pred)
    known_macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    known_weighted_f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

    covered = conf >= conf_threshold
    coverage = float(np.mean(covered))
    unknown_rate = 1.0 - coverage

    if np.any(covered):
        covered_accuracy = accuracy_score(y_test[covered], pred[covered])
        covered_macro_f1 = f1_score(
            y_test[covered],
            pred[covered],
            average="macro",
            zero_division=0,
        )
        covered_weighted_f1 = f1_score(
            y_test[covered],
            pred[covered],
            average="weighted",
            zero_division=0,
        )
    else:
        covered_accuracy = 0.0
        covered_macro_f1 = 0.0
        covered_weighted_f1 = 0.0

    summary_rows = [{
        "dataset": "MIT-BIH",
        "task": "Stage 2: Abnormal subtype classification",
        "model": "MIT-BIH Stage 2 Subtypes S/V/F v2",
        "samples": int(len(y_test)),
        "known_accuracy_without_rejection": float(known_accuracy),
        "known_macro_f1_without_rejection": float(known_macro_f1),
        "known_weighted_f1_without_rejection": float(known_weighted_f1),
        "confidence_threshold": float(conf_threshold),
        "coverage": float(coverage),
        "unknown_rate": float(unknown_rate),
        "covered_accuracy": float(covered_accuracy),
        "covered_macro_f1": float(covered_macro_f1),
        "covered_weighted_f1": float(covered_weighted_f1),
    }]

    report = classification_report(
        y_test,
        pred,
        labels=[0, 1, 2],
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for label in label_names:
        per_class_rows.append({
            "class": label,
            "class_name": display_names[label],
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1_score": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        })

    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
    confusion_rows = []
    for i, true_label in enumerate(label_names):
        row = {"true_class": true_label}
        for j, pred_label in enumerate(label_names):
            row[f"pred_{pred_label}"] = int(cm[i, j])
        confusion_rows.append(row)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(per_class_rows),
        pd.DataFrame(confusion_rows),
    )


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing Dataset/ and Models/",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    reports_dir = project_root / "Reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", project_root)
    print("Reports dir:", reports_dir)

    ptbdb_results, ptbdb_cm = evaluate_ptbdb(project_root, batch_size=args.batch_size)
    mit1_results, mit1_cm = evaluate_mitbih_stage1(project_root, batch_size=args.batch_size)
    mit2_summary, mit2_per_class, mit2_cm = evaluate_mitbih_stage2(project_root, batch_size=args.batch_size)

    # Combined model summary table
    combined_rows = []

    if not ptbdb_results.empty:
        final_live = ptbdb_results[
            ptbdb_results["model"] == "Final Live Pipeline: TCN-first Conditional Ensemble"
        ]
        if not final_live.empty:
            r = final_live.iloc[0].to_dict()
            combined_rows.append({
                "section": "Live Monitoring",
                "dataset": r["dataset"],
                "task": r["task"],
                "model": r["model"],
                "samples": r["samples"],
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1_score": r["f1_score"],
                "notes": "Main live normal/abnormal classifier",
            })

    if not mit1_results.empty:
        r = mit1_results.iloc[0].to_dict()
        combined_rows.append({
            "section": "Post-session Analysis",
            "dataset": r["dataset"],
            "task": r["task"],
            "model": r["model"],
            "samples": r["samples"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"],
            "notes": "Filters normal-like vs abnormal beats before subtype analysis",
        })

    if not mit2_summary.empty:
        r = mit2_summary.iloc[0].to_dict()
        combined_rows.append({
            "section": "Post-session Analysis",
            "dataset": r["dataset"],
            "task": r["task"],
            "model": r["model"],
            "samples": r["samples"],
            "accuracy": r["known_accuracy_without_rejection"],
            "precision": "",
            "recall": "",
            "f1_score": r["known_macro_f1_without_rejection"],
            "notes": f"Subtype model with confidence rejection; unknown rate = {r['unknown_rate']:.3f}",
        })

    combined = pd.DataFrame(combined_rows)

    # Save CSV files
    ptbdb_results.to_csv(reports_dir / "chapter12_ptbdb_model_comparison.csv", index=False)
    ptbdb_cm.to_csv(reports_dir / "chapter12_ptbdb_confusion_matrix.csv", index=False)

    mit1_results.to_csv(reports_dir / "chapter12_mitbih_stage1_results.csv", index=False)
    mit1_cm.to_csv(reports_dir / "chapter12_mitbih_stage1_confusion_matrix.csv", index=False)

    mit2_summary.to_csv(reports_dir / "chapter12_mitbih_stage2_summary.csv", index=False)
    mit2_per_class.to_csv(reports_dir / "chapter12_mitbih_stage2_per_class.csv", index=False)
    mit2_cm.to_csv(reports_dir / "chapter12_mitbih_stage2_confusion_matrix.csv", index=False)

    combined.to_csv(reports_dir / "chapter12_model_results_summary.csv", index=False)

    # Save Excel workbook
    excel_path = reports_dir / "chapter12_model_evaluation_tables.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Summary", index=False)
        ptbdb_results.to_excel(writer, sheet_name="PTBDB Models", index=False)
        ptbdb_cm.to_excel(writer, sheet_name="PTBDB CM", index=False)
        mit1_results.to_excel(writer, sheet_name="MITBIH Stage1", index=False)
        mit1_cm.to_excel(writer, sheet_name="MITBIH Stage1 CM", index=False)
        mit2_summary.to_excel(writer, sheet_name="MITBIH Stage2", index=False)
        mit2_per_class.to_excel(writer, sheet_name="MITBIH Stage2 Classes", index=False)
        mit2_cm.to_excel(writer, sheet_name="MITBIH Stage2 CM", index=False)

    print("\nDone. Files saved:")
    print(reports_dir / "chapter12_model_results_summary.csv")
    print(reports_dir / "chapter12_ptbdb_model_comparison.csv")
    print(reports_dir / "chapter12_mitbih_stage1_results.csv")
    print(reports_dir / "chapter12_mitbih_stage2_summary.csv")
    print(excel_path)

    print("\nMain summary table:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()