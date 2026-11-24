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


@dataclass
class ThresholdResult:
    threshold: float
    accuracy: float
    precision_abnormal: float
    recall_abnormal: float
    f1_abnormal: float
    specificity_normal: float
    roc_auc: float
    tp: int
    tn: int
    fp: int
    fn: int
    confusion_matrix: list[list[int]]
    classification_report: dict


def load_mitbih_test(dataset_dir: Path):
    path = dataset_dir / "mitbih_test.csv"

    if not path.exists():
        raise FileNotFoundError("Missing Dataset/mitbih_test.csv")

    df = pd.read_csv(path, header=None)

    if df.shape[1] < 188:
        raise ValueError("mitbih_test.csv must have 187 features + 1 label column.")

    x = df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, 187].to_numpy(dtype=np.int64)

    # MIT-BIH:
    # 0 = Normal
    # 1,2,3,4 = Abnormal
    y_bin = (y_raw != 0).astype(np.int64)

    return x, y_bin


def z_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> ThresholdResult:
    y_pred = (y_prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    auc = roc_auc_score(y_true, y_prob)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    return ThresholdResult(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_abnormal=float(precision_score(y_true, y_pred, zero_division=0)),
        recall_abnormal=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_abnormal=float(f1_score(y_true, y_pred, zero_division=0)),
        specificity_normal=float(specificity),
        roc_auc=float(auc),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
    )


def main():
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"
    reports_dir = project_root / "Reports"
    reports_dir.mkdir(exist_ok=True)

    model_path = models_dir / "mitbih_stage1_binary.keras"

    if not model_path.exists():
        raise FileNotFoundError("Missing Models/mitbih_stage1_binary.keras")

    print("Loading MIT-BIH test data...")
    x, y = load_mitbih_test(dataset_dir)

    print("Normal samples:", int((y == 0).sum()))
    print("Abnormal samples:", int((y == 1).sum()))

    x = z_norm(x).reshape(-1, 187, 1)

    print("Loading Stage 1 model...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Predicting probabilities...")
    y_prob = model.predict(x, verbose=0, batch_size=256).reshape(-1)

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.76,
        0.80,
    ]

    results: list[ThresholdResult] = []

    print("\n================ THRESHOLD RESULTS ================")

    for thr in thresholds:
        r = compute_metrics(y, y_prob, threshold=thr)
        results.append(r)

        print(
            f"thr={r.threshold:.2f} | "
            f"acc={r.accuracy:.4f} | "
            f"prec_abn={r.precision_abnormal:.4f} | "
            f"rec_abn={r.recall_abnormal:.4f} | "
            f"f1_abn={r.f1_abnormal:.4f} | "
            f"spec={r.specificity_normal:.4f} | "
            f"FP={r.fp} | FN={r.fn}"
        )

    # Selection logic:
    # We prefer recall >= 0.60 if possible, then best abnormal F1.
    candidates = [r for r in results if r.recall_abnormal >= 0.60]

    if candidates:
        best = max(candidates, key=lambda r: r.f1_abnormal)
    else:
        best = max(results, key=lambda r: r.f1_abnormal)

    print("\n================ BEST SUGGESTED THRESHOLD ================")
    print(
        f"Best threshold: {best.threshold:.2f}\n"
        f"Accuracy: {best.accuracy:.4f}\n"
        f"Precision abnormal: {best.precision_abnormal:.4f}\n"
        f"Recall abnormal: {best.recall_abnormal:.4f}\n"
        f"F1 abnormal: {best.f1_abnormal:.4f}\n"
        f"Specificity normal: {best.specificity_normal:.4f}\n"
        f"FP: {best.fp}\n"
        f"FN: {best.fn}\n"
        f"Confusion matrix [[TN, FP], [FN, TP]]: {best.confusion_matrix}"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"mitbih_stage1_threshold_tuning_{stamp}.json"
    csv_path = reports_dir / f"mitbih_stage1_threshold_tuning_{stamp}.csv"

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path),
        "dataset": "MIT-BIH test binary: Normal vs Abnormal",
        "selection_rule": "Prefer recall_abnormal >= 0.60, then highest f1_abnormal. If none, choose highest f1_abnormal.",
        "best_suggested_threshold": asdict(best),
        "results": [asdict(r) for r in results],
    }

    json_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "threshold",
                "accuracy",
                "precision_abnormal",
                "recall_abnormal",
                "f1_abnormal",
                "specificity_normal",
                "roc_auc",
                "tp",
                "tn",
                "fp",
                "fn",
                "confusion_matrix",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r.threshold,
                    r.accuracy,
                    r.precision_abnormal,
                    r.recall_abnormal,
                    r.f1_abnormal,
                    r.specificity_normal,
                    r.roc_auc,
                    r.tp,
                    r.tn,
                    r.fp,
                    r.fn,
                    json.dumps(r.confusion_matrix),
                ]
            )

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()