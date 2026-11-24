from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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
        cfg.update({"length": self.length, "d_model": self.d_model})
        return cfg


@dataclass
class Result:
    model_name: str
    threshold: float
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    auc: float
    tp: int
    tn: int
    fp: int
    fn: int
    confusion_matrix: list[list[int]]
    classification_report: dict


def load_keras_model(path: Path):
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={"PositionalEncoding": PositionalEncoding},
    )


def load_ptbdb(dataset_dir: Path):
    normal_path = dataset_dir / "ptbdb_normal.csv"
    abnormal_path = dataset_dir / "ptbdb_abnormal.csv"

    if not normal_path.exists() or not abnormal_path.exists():
        raise FileNotFoundError(
            "Missing Dataset/ptbdb_normal.csv or Dataset/ptbdb_abnormal.csv"
        )

    df_n = pd.read_csv(normal_path, header=None)
    df_a = pd.read_csv(abnormal_path, header=None)

    x_n = df_n.iloc[:, :187].to_numpy(dtype=np.float32)
    x_a = df_a.iloc[:, :187].to_numpy(dtype=np.float32)

    y_n = np.zeros(len(x_n), dtype=np.int64)
    y_a = np.ones(len(x_a), dtype=np.int64)

    x = np.vstack([x_n, x_a])
    y = np.concatenate([y_n, y_a])

    return x, y


def normalize_ptbdb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x, axis=1, keepdims=True)
    x = x - med
    scale = np.max(np.abs(x), axis=1, keepdims=True) + 1e-9
    return x / scale


def predict_binary(model, x: np.ndarray) -> np.ndarray:
    x3 = x.reshape(-1, 187, 1).astype(np.float32)
    return model.predict(x3, verbose=0, batch_size=256).reshape(-1)


def compute_result(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.60,
) -> Result:
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    return Result(
        model_name=name,
        threshold=float(threshold),
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        specificity=float(specificity),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=float(auc),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
    )


def weighted_average(preds: list[tuple[float, np.ndarray]]) -> np.ndarray:
    total_w = sum(w for w, _ in preds)
    return sum(w * p for w, p in preds) / total_w


def main():
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"
    reports_dir = project_root / "Reports"
    reports_dir.mkdir(exist_ok=True)

    print("Loading PTBDB...")
    x, y = load_ptbdb(dataset_dir)
    x = normalize_ptbdb(x)

    print("Loading models...")
    resnet = load_keras_model(models_dir / "ptbdb_resnet1d_fix.keras")
    tcn = load_keras_model(models_dir / "ptbdb_tcn.keras")
    transformer = load_keras_model(models_dir / "ptbdb_transformer_fix.keras")

    print("Predicting ResNet...")
    p_resnet = predict_binary(resnet, x)

    print("Predicting TCN...")
    p_tcn = predict_binary(tcn, x)

    print("Predicting Transformer...")
    p_transformer = predict_binary(transformer, x)

    results: list[Result] = []

    # 1) Individual models
    results.append(compute_result("ResNet only", y, p_resnet, threshold=0.60))
    results.append(compute_result("TCN only", y, p_tcn, threshold=0.60))
    results.append(compute_result("Transformer only", y, p_transformer, threshold=0.60))

    # 2) Current pipeline: ResNet first, uncertain only [0.30, 0.70]
    p_current = p_resnet.copy()
    mask = (p_resnet >= 0.30) & (p_resnet <= 0.70)
    p_current[mask] = weighted_average([
        (0.50, p_resnet[mask]),
        (0.30, p_tcn[mask]),
        (0.20, p_transformer[mask]),
    ])
    results.append(compute_result("Current ResNet-first ensemble [0.30,0.70]", y, p_current, 0.60))

    # 3) ResNet first, but wider uncertainty range
    p_resnet_wide = p_resnet.copy()
    mask = (p_resnet >= 0.20) & (p_resnet <= 0.85)
    p_resnet_wide[mask] = weighted_average([
        (0.50, p_resnet[mask]),
        (0.30, p_tcn[mask]),
        (0.20, p_transformer[mask]),
    ])
    results.append(compute_result("ResNet-first wide ensemble [0.20,0.85]", y, p_resnet_wide, 0.60))

    # 4) TCN first, uncertain only [0.30, 0.70]
    p_tcn_first = p_tcn.copy()
    mask = (p_tcn >= 0.30) & (p_tcn <= 0.70)
    p_tcn_first[mask] = weighted_average([
        (0.50, p_tcn[mask]),
        (0.30, p_resnet[mask]),
        (0.20, p_transformer[mask]),
    ])
    results.append(compute_result("TCN-first ensemble [0.30,0.70]", y, p_tcn_first, 0.60))

    # 5) TCN first, wider uncertainty range
    p_tcn_first_wide = p_tcn.copy()
    mask = (p_tcn >= 0.20) & (p_tcn <= 0.85)
    p_tcn_first_wide[mask] = weighted_average([
        (0.50, p_tcn[mask]),
        (0.30, p_resnet[mask]),
        (0.20, p_transformer[mask]),
    ])
    results.append(compute_result("TCN-first wide ensemble [0.20,0.85]", y, p_tcn_first_wide, 0.60))

    # Print results
    print("\n================ RESULTS ================")
    for r in results:
        print(
            f"{r.model_name}: "
            f"ACC={r.accuracy:.4f} | "
            f"PREC={r.precision:.4f} | "
            f"REC={r.recall:.4f} | "
            f"SPEC={r.specificity:.4f} | "
            f"F1={r.f1:.4f} | "
            f"AUC={r.auc:.4f} | "
            f"FP={r.fp} | FN={r.fn}"
        )

    # Save JSON
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = reports_dir / f"ptbdb_experiments_{stamp}.json"
    csv_path = reports_dir / f"ptbdb_experiments_{stamp}.csv"

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": [asdict(r) for r in results],
    }

    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name",
            "threshold",
            "n_samples",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc",
            "tp",
            "tn",
            "fp",
            "fn",
            "confusion_matrix",
        ])

        for r in results:
            writer.writerow([
                r.model_name,
                r.threshold,
                r.n_samples,
                r.accuracy,
                r.precision,
                r.recall,
                r.specificity,
                r.f1,
                r.auc,
                r.tp,
                r.tn,
                r.fp,
                r.fn,
                json.dumps(r.confusion_matrix),
            ])

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
    