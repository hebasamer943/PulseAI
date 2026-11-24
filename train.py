from _future_ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier

CLASS_NAMES: Dict[int, str] = {
    0: "N - Normal",
    1: "S - Supraventricular",
    2: "V - Ventricular",
    3: "F - Fusion",
    4: "Q - Unknown",
}


def load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, :-1].astype(np.float32).values
    y = df.iloc[:, -1].astype(int).values
    return x, y


def balance_data(x: np.ndarray, y: np.ndarray, target_per_class: int = 30000) -> Tuple[np.ndarray, np.ndarray]:
    frames = []
    df = pd.DataFrame(x)
    df["target"] = y

    for class_id in sorted(df["target"].unique()):
        class_df = df[df["target"] == class_id]
        n = len(class_df)
        if n > target_per_class:
            sampled = resample(class_df, replace=False, n_samples=target_per_class, random_state=42)
        elif n < target_per_class:
            sampled = resample(class_df, replace=True, n_samples=target_per_class, random_state=42)
        else:
            sampled = class_df
        frames.append(sampled)

    balanced = pd.concat(frames).sample(frac=1.0, random_state=42).reset_index(drop=True)
    x_bal = balanced.drop(columns=["target"]).values.astype(np.float32)
    y_bal = balanced["target"].values.astype(int)
    return x_bal, y_bal


def save_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    fig.colorbar(im)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([CLASS_NAMES[i] for i in range(len(CLASS_NAMES))], rotation=25, ha="right")
    ax.set_yticklabels([CLASS_NAMES[i] for i in range(len(CLASS_NAMES))])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - MIT-BIH")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIT-BIH arrhythmia classifier")
    parser.add_argument("--train_csv", default="data/mitbih_train.csv")
    parser.add_argument("--test_csv", default="data/mitbih_test.csv")
    parser.add_argument("--target_per_class", type=int, default=30000)
    parser.add_argument("--debug_rows", type=int, default=0, help="Use only first N rows from train for quick test")
    args = parser.parse_args()

    project_root = Path(_file_).resolve().parents[1]
    train_path = project_root / args.train_csv
    test_path = project_root / args.test_csv
    model_dir = project_root / "models"
    reports_dir = project_root / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    x_train, y_train = load_csv(train_path)
    x_test, y_test = load_csv(test_path)

    if args.debug_rows > 0:
        x_train = x_train[: args.debug_rows]
        y_train = y_train[: args.debug_rows]
        print(f"Debug mode: using first {args.debug_rows} train rows")

    print(f"Original train shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")

    print("Balancing train data...")
    x_bal, y_bal = balance_data(x_train, y_train, target_per_class=args.target_per_class)
    print(f"Balanced train shape: {x_bal.shape}")

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_bal,
        y_bal,
        test_size=0.15,
        random_state=42,
        stratify=y_bal,
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        n_estimators=350,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print("Training model...")
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    print("Evaluating...")
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "class_names": CLASS_NAMES,
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=[CLASS_NAMES[i] for i in range(5)],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    model_path = model_dir / "mitbih_xgb_model.joblib"
    metrics_path = reports_dir / "metrics.json"
    report_path = reports_dir / "classification_report.json"
    cm_path = reports_dir / "confusion_matrix.png"

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    save_confusion_matrix(cm, cm_path)

    print("\nTraining complete.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print("\nFinal metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}")


if _name_ == "_main_":
    main()