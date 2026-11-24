from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Custom layer needed for loading Transformer model
# ============================================================

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, length=187, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model

        pos = np.arange(length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(
            10000,
            (2 * (i // 2)) / np.float32(d_model),
        )
        angles = pos * angle_rates

        pe = np.zeros((length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :])

    def call(self, x):
        return x + self.pe[:, : tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "length": self.length,
                "d_model": self.d_model,
            }
        )
        return cfg


# ============================================================
# Metrics containers
# ============================================================

@dataclass
class BinaryResult:
    model_name: str
    dataset: str
    threshold: float
    n_samples: int
    accuracy: float
    precision: float
    recall_sensitivity: float
    specificity: float
    f1_score: float
    roc_auc: float | None
    tp: int
    tn: int
    fp: int
    fn: int
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]


@dataclass
class MultiClassResult:
    model_name: str
    dataset: str
    n_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]


# ============================================================
# Basic helpers
# ============================================================

def ensure_187(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size > 187:
        x = x[:187]
    if x.size < 187:
        x = np.pad(x, (0, 187 - x.size))
    return x.astype(np.float32)


def normalize_median_maxabs(x: np.ndarray) -> np.ndarray:
    """
    Same style used by the live PTBDB pipeline:
    median center + max absolute scaling.
    """
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x, axis=1, keepdims=True)
    x = x - med
    scale = np.max(np.abs(x), axis=1, keepdims=True) + 1e-9
    return x / scale


def normalize_zscore(x: np.ndarray) -> np.ndarray:
    """
    Used for MIT-BIH stage models.
    """
    x = np.asarray(x, dtype=np.float32)
    mu = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def load_keras_model(path: Path):
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={"PositionalEncoding": PositionalEncoding},
    )


def predict_binary(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    x3 = np.asarray(x, dtype=np.float32).reshape(-1, 187, 1)
    prob = model.predict(x3, verbose=0, batch_size=batch_size).reshape(-1)
    return prob.astype(np.float32)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return None

    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return None


def compute_binary_result(
    model_name: str,
    dataset_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> BinaryResult:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    return BinaryResult(
        model_name=model_name,
        dataset=dataset_name,
        threshold=float(threshold),
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_sensitivity=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(specificity),
        f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=safe_roc_auc(y_true, y_prob),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
    )


def compute_multiclass_result(
    model_name: str,
    dataset_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    target_names: list[str],
) -> MultiClassResult:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    return MultiClassResult(
        model_name=model_name,
        dataset=dataset_name,
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_precision=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        weighted_recall=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
    )


# ============================================================
# Dataset loaders
# ============================================================

def load_ptbdb(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    normal_path = dataset_dir / "ptbdb_normal.csv"
    abnormal_path = dataset_dir / "ptbdb_abnormal.csv"

    if not normal_path.exists() or not abnormal_path.exists():
        raise FileNotFoundError(
            "Missing PTBDB files. Expected Dataset/ptbdb_normal.csv and Dataset/ptbdb_abnormal.csv"
        )

    df_n = pd.read_csv(normal_path, header=None)
    df_a = pd.read_csv(abnormal_path, header=None)

    x_n = df_n.iloc[:, :187].to_numpy(dtype=np.float32)
    x_a = df_a.iloc[:, :187].to_numpy(dtype=np.float32)

    y_n = np.zeros(len(x_n), dtype=np.int64)
    y_a = np.ones(len(x_a), dtype=np.int64)

    x = np.vstack([x_n, x_a]).astype(np.float32)
    y = np.concatenate([y_n, y_a]).astype(np.int64)

    return x, y


def load_mitbih_test_binary(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_dir / "mitbih_test.csv"

    if not path.exists():
        raise FileNotFoundError("Missing Dataset/mitbih_test.csv")

    df = pd.read_csv(path, header=None)

    if df.shape[1] < 188:
        raise ValueError("mitbih_test.csv must have 187 features + label column.")

    x = df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, 187].to_numpy(dtype=np.int64)

    # MIT-BIH labels:
    # 0 = N
    # 1 = S
    # 2 = V
    # 3 = F
    # 4 = Q
    # Binary: anything non-zero = abnormal
    y_bin = (y_raw != 0).astype(np.int64)

    return x, y_bin


def load_mitbih_test_stage2_svf(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_dir / "mitbih_test.csv"

    if not path.exists():
        raise FileNotFoundError("Missing Dataset/mitbih_test.csv")

    df = pd.read_csv(path, header=None)

    if df.shape[1] < 188:
        raise ValueError("mitbih_test.csv must have 187 features + label column.")

    x_all = df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, 187].to_numpy(dtype=np.int64)

    # Stage 2 model only handles S / V / F.
    # Exclude N=0 and Q=4.
    mask = np.isin(y_raw, [1, 2, 3])
    x = x_all[mask]
    y_raw = y_raw[mask]

    # Map MIT-BIH labels to local stage2 labels:
    # 1(S) -> 0
    # 2(V) -> 1
    # 3(F) -> 2
    mapper = {1: 0, 2: 1, 3: 2}
    y = np.asarray([mapper[int(v)] for v in y_raw], dtype=np.int64)

    return x, y


# ============================================================
# Ensemble logic: same idea as app
# ResNet first, then TCN/Transformer only in uncertain zone
# ============================================================

def predict_ptbdb_ensemble(
    x_norm: np.ndarray,
    models_dir: Path,
    batch_size: int = 256,
    unc_low: float = 0.30,
    unc_high: float = 0.70,
    w_resnet: float = 0.50,
    w_tcn: float = 0.30,
    w_transformer: float = 0.20,
) -> np.ndarray:
    resnet_path = models_dir / "ptbdb_resnet1d_fix.keras"
    tcn_path = models_dir / "ptbdb_tcn.keras"
    transformer_path = models_dir / "ptbdb_transformer_fix.keras"

    if not resnet_path.exists():
        raise FileNotFoundError("Missing ptbdb_resnet1d_fix.keras, cannot evaluate ensemble.")

    resnet = load_keras_model(resnet_path)
    p_res = predict_binary(resnet, x_norm, batch_size=batch_size)

    final = p_res.copy()
    uncertain_mask = (p_res >= unc_low) & (p_res <= unc_high)

    if not np.any(uncertain_mask):
        return final

    weighted_parts = [(w_resnet, p_res[uncertain_mask])]

    if tcn_path.exists():
        tcn = load_keras_model(tcn_path)
        p_tcn = predict_binary(tcn, x_norm[uncertain_mask], batch_size=batch_size)
        weighted_parts.append((w_tcn, p_tcn))

    if transformer_path.exists():
        transformer = load_keras_model(transformer_path)
        p_tr = predict_binary(transformer, x_norm[uncertain_mask], batch_size=batch_size)
        weighted_parts.append((w_transformer, p_tr))

    total_w = sum(w for w, _ in weighted_parts)
    p_uncertain = sum(w * p for w, p in weighted_parts) / max(total_w, 1e-9)

    final[uncertain_mask] = p_uncertain
    return final.astype(np.float32)


# ============================================================
# Main evaluation
# ============================================================

def evaluate_all(project_root: Path) -> tuple[list[BinaryResult], list[MultiClassResult]]:
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"

    binary_results: list[BinaryResult] = []
    multiclass_results: list[MultiClassResult] = []

    print("\n========================================")
    print("Loading datasets")
    print("========================================")

    ptbdb_x, ptbdb_y = load_ptbdb(dataset_dir)
    ptbdb_x_norm = normalize_median_maxabs(ptbdb_x)

    print(f"PTBDB: X={ptbdb_x.shape}, y={ptbdb_y.shape}")

    try:
        mit_x_bin, mit_y_bin = load_mitbih_test_binary(dataset_dir)
        mit_x_bin_z = normalize_zscore(mit_x_bin)
        print(f"MIT-BIH binary: X={mit_x_bin.shape}, y={mit_y_bin.shape}")
    except Exception as e:
        mit_x_bin = mit_y_bin = mit_x_bin_z = None
        print(f"Skipping MIT-BIH binary dataset: {e}")

    try:
        mit_x_svf, mit_y_svf = load_mitbih_test_stage2_svf(dataset_dir)
        mit_x_svf_z = normalize_zscore(mit_x_svf)
        print(f"MIT-BIH S/V/F: X={mit_x_svf.shape}, y={mit_y_svf.shape}")
    except Exception as e:
        mit_x_svf = mit_y_svf = mit_x_svf_z = None
        print(f"Skipping MIT-BIH stage2 dataset: {e}")

    print("\n========================================")
    print("Evaluating PTBDB binary models")
    print("========================================")

    ptbdb_models = [
        ("PTBDB ResNet1D", "ptbdb_resnet1d_fix.keras", 0.60),
        ("PTBDB TCN", "ptbdb_tcn.keras", 0.60),
        ("PTBDB Transformer", "ptbdb_transformer_fix.keras", 0.60),
    ]

    for model_title, filename, threshold in ptbdb_models:
        path = models_dir / filename

        if not path.exists():
            print(f"SKIP: {filename} not found.")
            continue

        print(f"Evaluating {model_title}...")
        model = load_keras_model(path)
        prob = predict_binary(model, ptbdb_x_norm)

        result = compute_binary_result(
            model_name=model_title,
            dataset_name="PTBDB normal/abnormal",
            y_true=ptbdb_y,
            y_prob=prob,
            threshold=threshold,
        )
        binary_results.append(result)
        print_binary_summary(result)

    print("\n========================================")
    print("Evaluating PTBDB ensemble pipeline")
    print("========================================")

    try:
        prob_ens = predict_ptbdb_ensemble(ptbdb_x_norm, models_dir)

        result = compute_binary_result(
            model_name="PTBDB Ensemble Pipeline",
            dataset_name="PTBDB normal/abnormal",
            y_true=ptbdb_y,
            y_prob=prob_ens,
            threshold=0.60,
        )
        binary_results.append(result)
        print_binary_summary(result)

    except Exception as e:
        print(f"SKIP ensemble: {e}")

    print("\n========================================")
    print("Evaluating image/inception model if available")
    print("========================================")

    image_path = models_dir / "mitbih_binary_inception.keras"

    if image_path.exists() and mit_x_bin_z is not None and mit_y_bin is not None:
        try:
            print("Evaluating MIT-BIH Image/Inception Binary...")
            image_model = load_keras_model(image_path)
            prob = predict_binary(image_model, mit_x_bin_z)

            result = compute_binary_result(
                model_name="MIT-BIH Image/Inception Binary",
                dataset_name="MIT-BIH test binary",
                y_true=mit_y_bin,
                y_prob=prob,
                threshold=0.60,
            )
            binary_results.append(result)
            print_binary_summary(result)

        except Exception as e:
            print(f"SKIP image/inception: {e}")
    else:
        print("SKIP image/inception: model or MIT-BIH binary dataset not found.")

    print("\n========================================")
    print("Evaluating MIT-BIH Stage 1 binary model")
    print("========================================")

    stage1_path = models_dir / "mitbih_stage1_binary.keras"

    if stage1_path.exists() and mit_x_bin_z is not None and mit_y_bin is not None:
        try:
            print("Evaluating MIT-BIH Stage 1 Binary...")
            stage1 = load_keras_model(stage1_path)
            prob = predict_binary(stage1, mit_x_bin_z)

            threshold = load_stage1_threshold(models_dir)

            result = compute_binary_result(
                model_name="MIT-BIH Stage 1 Binary",
                dataset_name="MIT-BIH test normal-vs-abnormal",
                y_true=mit_y_bin,
                y_prob=prob,
                threshold=threshold,
            )
            binary_results.append(result)
            print_binary_summary(result)

        except Exception as e:
            print(f"SKIP MIT-BIH Stage 1: {e}")
    else:
        print("SKIP MIT-BIH Stage 1: model or dataset not found.")

    print("\n========================================")
    print("Evaluating MIT-BIH Stage 2 subtype model")
    print("========================================")

    stage2_path = models_dir / "mitbih_stage2_subtypes_svf.keras"

    if stage2_path.exists() and mit_x_svf_z is not None and mit_y_svf is not None:
        try:
            print("Evaluating MIT-BIH Stage 2 S/V/F...")
            stage2 = load_keras_model(stage2_path)

            prob = stage2.predict(
                mit_x_svf_z.reshape(-1, 187, 1),
                verbose=0,
                batch_size=256,
            )
            pred = np.argmax(prob, axis=1).astype(np.int64)

            result = compute_multiclass_result(
                model_name="MIT-BIH Stage 2 Subtypes S/V/F",
                dataset_name="MIT-BIH test abnormal subtypes only",
                y_true=mit_y_svf,
                y_pred=pred,
                labels=[0, 1, 2],
                target_names=["S", "V", "F"],
            )
            multiclass_results.append(result)
            print_multiclass_summary(result)

        except Exception as e:
            print(f"SKIP MIT-BIH Stage 2: {e}")
    else:
        print("SKIP MIT-BIH Stage 2: model or dataset not found.")

    return binary_results, multiclass_results


def load_stage1_threshold(models_dir: Path) -> float:
    """
    If metadata exists, try to use a threshold from it.
    Otherwise use 0.50.
    """
    meta_path = models_dir / "mitbih_stage1_binary_meta.json"

    if not meta_path.exists():
        return 0.50

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.50

    for key in ["best_threshold_from_val", "threshold"]:
        val = data.get(key)
        if isinstance(val, (int, float)):
            return float(val)

    band = data.get("uncertainty_band", {})
    if isinstance(band, dict):
        high = band.get("abnormal_if_prob_abnormal_ge")
        if isinstance(high, (int, float)):
            return float(high)

    return 0.50


# ============================================================
# Saving output
# ============================================================

def print_binary_summary(r: BinaryResult) -> None:
    auc = "None" if r.roc_auc is None else f"{r.roc_auc:.4f}"

    print(
        f"{r.model_name}: "
        f"ACC={r.accuracy:.4f}, "
        f"PREC={r.precision:.4f}, "
        f"REC={r.recall_sensitivity:.4f}, "
        f"SPEC={r.specificity:.4f}, "
        f"F1={r.f1_score:.4f}, "
        f"AUC={auc}, "
        f"THR={r.threshold:.2f}"
    )
    print(f"CM [[TN, FP], [FN, TP]] = {r.confusion_matrix}")


def print_multiclass_summary(r: MultiClassResult) -> None:
    print(
        f"{r.model_name}: "
        f"ACC={r.accuracy:.4f}, "
        f"MacroF1={r.macro_f1:.4f}, "
        f"WeightedF1={r.weighted_f1:.4f}"
    )
    print(f"CM = {r.confusion_matrix}")


def save_outputs(
    project_root: Path,
    binary_results: list[BinaryResult],
    multiclass_results: list[MultiClassResult],
) -> None:
    reports_dir = project_root / "Reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"all_model_metrics_{stamp}.json"
    csv_path = reports_dir / f"all_model_metrics_{stamp}.csv"

    full = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "binary_results": [asdict(r) for r in binary_results],
        "multiclass_results": [asdict(r) for r in multiclass_results],
    }

    json_path.write_text(
        json.dumps(full, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "model_name",
                "dataset",
                "task_type",
                "threshold",
                "n_samples",
                "accuracy",
                "precision",
                "recall_sensitivity",
                "specificity",
                "f1_score",
                "roc_auc",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
                "tp",
                "tn",
                "fp",
                "fn",
                "confusion_matrix",
            ]
        )

        for r in binary_results:
            writer.writerow(
                [
                    r.model_name,
                    r.dataset,
                    "binary",
                    r.threshold,
                    r.n_samples,
                    r.accuracy,
                    r.precision,
                    r.recall_sensitivity,
                    r.specificity,
                    r.f1_score,
                    r.roc_auc,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    r.tp,
                    r.tn,
                    r.fp,
                    r.fn,
                    json.dumps(r.confusion_matrix),
                ]
            )

        for r in multiclass_results:
            writer.writerow(
                [
                    r.model_name,
                    r.dataset,
                    "multiclass",
                    "",
                    r.n_samples,
                    r.accuracy,
                    "",
                    "",
                    "",
                    "",
                    "",
                    r.macro_precision,
                    r.macro_recall,
                    r.macro_f1,
                    r.weighted_precision,
                    r.weighted_recall,
                    r.weighted_f1,
                    "",
                    "",
                    "",
                    "",
                    json.dumps(r.confusion_matrix),
                ]
            )

    print("\n========================================")
    print("Saved reports")
    print("========================================")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


def main():
    project_root = Path(__file__).resolve().parent

    print("\nProject root:", project_root)

    binary_results, multiclass_results = evaluate_all(project_root)
    save_outputs(project_root, binary_results, multiclass_results)

    print("\nDone.")


if __name__ == "__main__":
    main()