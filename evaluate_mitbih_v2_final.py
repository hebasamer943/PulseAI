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


MIT_TO_LOCAL = {1: 0, 2: 1, 3: 2}
LOCAL_TO_LABEL = {0: "S", 1: "V", 2: "F"}


@dataclass
class Stage1Result:
    model_name: str
    threshold: float
    n_samples: int
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


@dataclass
class Stage2Result:
    model_name: str
    n_samples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion_matrix: list[list[int]]
    classification_report: dict


@dataclass
class Stage2RejectionResult:
    model_name: str
    confidence_threshold: float
    n_samples: int
    covered_samples: int
    unknown_samples: int
    coverage: float
    unknown_rate: float
    covered_accuracy: float
    covered_macro_f1: float
    covered_weighted_f1: float
    covered_confusion_matrix: list[list[int]]
    covered_classification_report: dict


@dataclass
class TwoStagePipelineResult:
    stage1_threshold: float
    stage2_confidence_threshold: float
    total_samples: int
    normal_samples: int
    abnormal_svf_samples: int
    q_unknown_samples: int

    stage1_binary_accuracy: float
    stage1_abnormal_precision: float
    stage1_abnormal_recall: float
    stage1_abnormal_f1: float

    final_5class_accuracy_known_labels: float
    final_macro_f1_known_labels: float
    final_weighted_f1_known_labels: float
    final_confusion_matrix_known_labels: list[list[int]]
    final_classification_report_known_labels: dict

    subtype_coverage_on_true_svf: float
    subtype_unknown_rate_on_true_svf: float


def z_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def load_test_full(dataset_dir: Path):
    path = dataset_dir / "mitbih_test.csv"

    if not path.exists():
        raise FileNotFoundError("Missing Dataset/mitbih_test.csv")

    df = pd.read_csv(path, header=None)

    x = df.iloc[:, :187].to_numpy(dtype=np.float32)
    y_raw = df.iloc[:, 187].to_numpy(dtype=np.int64)

    return x, y_raw


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return tf.keras.models.load_model(path, compile=False)


def load_threshold_from_meta(meta_path: Path, default: float) -> float:
    if not meta_path.exists():
        return default

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return default

    val = data.get("best_threshold_from_val")
    if isinstance(val, (int, float)):
        return float(val)

    val = data.get("recommended_threshold")
    if isinstance(val, (int, float)):
        return float(val)

    return default


def load_conf_threshold_from_meta(meta_path: Path, default: float) -> float:
    if not meta_path.exists():
        return default

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return default

    val = data.get("recommended_confidence_threshold")
    if isinstance(val, (int, float)):
        return float(val)

    return default


def evaluate_stage1(
    model,
    x_test: np.ndarray,
    y_raw: np.ndarray,
    threshold: float,
) -> Stage1Result:
    y_true = (y_raw != 0).astype(np.int64)

    x = z_norm(x_test).reshape(-1, 187, 1)
    prob = model.predict(x, verbose=0, batch_size=256).reshape(-1)

    pred = (prob >= threshold).astype(np.int64)

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    report = classification_report(
        y_true,
        pred,
        labels=[0, 1],
        target_names=["Normal", "Abnormal"],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    return Stage1Result(
        model_name="MIT-BIH Stage 1 v2",
        threshold=float(threshold),
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, pred)),
        precision_abnormal=float(precision_score(y_true, pred, zero_division=0)),
        recall_abnormal=float(recall_score(y_true, pred, zero_division=0)),
        f1_abnormal=float(f1_score(y_true, pred, zero_division=0)),
        specificity_normal=float(specificity),
        roc_auc=float(roc_auc_score(y_true, prob)),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
    )


def evaluate_stage2_known(
    model,
    x_test: np.ndarray,
    y_raw: np.ndarray,
) -> Stage2Result:
    # Only S/V/F, excluding N=0 and Q=4
    mask = np.isin(y_raw, [1, 2, 3])

    x = x_test[mask]
    y = y_raw[mask]

    y_true = np.asarray([MIT_TO_LOCAL[int(v)] for v in y], dtype=np.int64)

    x = z_norm(x).reshape(-1, 187, 1)
    probs = model.predict(x, verbose=0, batch_size=256)
    pred = np.argmax(probs, axis=1).astype(np.int64)

    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])

    report = classification_report(
        y_true,
        pred,
        labels=[0, 1, 2],
        target_names=["S", "V", "F"],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    return Stage2Result(
        model_name="MIT-BIH Stage 2 v2 S/V/F",
        n_samples=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, pred)),
        macro_precision=float(precision_score(y_true, pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, pred, average="macro", zero_division=0)),
        macro_f1=float(f1_score(y_true, pred, average="macro", zero_division=0)),
        weighted_precision=float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        weighted_recall=float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        weighted_f1=float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
    )


def evaluate_stage2_rejection(
    model,
    x_test: np.ndarray,
    y_raw: np.ndarray,
    confidence_threshold: float,
) -> Stage2RejectionResult:
    mask = np.isin(y_raw, [1, 2, 3])

    x = x_test[mask]
    y = y_raw[mask]

    y_true = np.asarray([MIT_TO_LOCAL[int(v)] for v in y], dtype=np.int64)

    x = z_norm(x).reshape(-1, 187, 1)
    probs = model.predict(x, verbose=0, batch_size=256)

    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1).astype(np.int64)

    covered = conf >= confidence_threshold

    covered_samples = int(np.sum(covered))
    unknown_samples = int(len(y_true) - covered_samples)
    coverage = covered_samples / len(y_true) if len(y_true) else 0.0
    unknown_rate = 1.0 - coverage

    if covered_samples == 0:
        return Stage2RejectionResult(
            model_name="MIT-BIH Stage 2 v2 with confidence rejection",
            confidence_threshold=float(confidence_threshold),
            n_samples=int(len(y_true)),
            covered_samples=0,
            unknown_samples=int(len(y_true)),
            coverage=0.0,
            unknown_rate=1.0,
            covered_accuracy=0.0,
            covered_macro_f1=0.0,
            covered_weighted_f1=0.0,
            covered_confusion_matrix=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            covered_classification_report={},
        )

    y_cov = y_true[covered]
    pred_cov = pred[covered]

    cm = confusion_matrix(y_cov, pred_cov, labels=[0, 1, 2])

    report = classification_report(
        y_cov,
        pred_cov,
        labels=[0, 1, 2],
        target_names=["S", "V", "F"],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    return Stage2RejectionResult(
        model_name="MIT-BIH Stage 2 v2 with confidence rejection",
        confidence_threshold=float(confidence_threshold),
        n_samples=int(len(y_true)),
        covered_samples=covered_samples,
        unknown_samples=unknown_samples,
        coverage=float(coverage),
        unknown_rate=float(unknown_rate),
        covered_accuracy=float(accuracy_score(y_cov, pred_cov)),
        covered_macro_f1=float(f1_score(y_cov, pred_cov, average="macro", zero_division=0)),
        covered_weighted_f1=float(f1_score(y_cov, pred_cov, average="weighted", zero_division=0)),
        covered_confusion_matrix=cm.astype(int).tolist(),
        covered_classification_report=report,
    )


def evaluate_two_stage_pipeline(
    stage1_model,
    stage2_model,
    x_test: np.ndarray,
    y_raw: np.ndarray,
    stage1_threshold: float,
    stage2_confidence_threshold: float,
) -> TwoStagePipelineResult:
    """
    Final two-stage behavior on MIT-BIH test:

    True labels:
      0 = Normal
      1 = S
      2 = V
      3 = F
      4 = Q / Unknown

    Final predicted labels:
      0 = Normal
      1 = S
      2 = V
      3 = F
      4 = Q / Unknown

    Important:
    Stage 2 only runs if Stage 1 says abnormal.
    If Stage 2 confidence < threshold, output Q/Unknown.
    """

    x_norm = z_norm(x_test).reshape(-1, 187, 1)

    # Stage 1
    p_abn = stage1_model.predict(x_norm, verbose=0, batch_size=256).reshape(-1)
    pred_binary = (p_abn >= stage1_threshold).astype(np.int64)

    y_binary = (y_raw != 0).astype(np.int64)

    stage1_binary_accuracy = accuracy_score(y_binary, pred_binary)
    stage1_abnormal_precision = precision_score(y_binary, pred_binary, zero_division=0)
    stage1_abnormal_recall = recall_score(y_binary, pred_binary, zero_division=0)
    stage1_abnormal_f1 = f1_score(y_binary, pred_binary, zero_division=0)

    # Final predictions: default Normal
    final_pred = np.zeros(len(y_raw), dtype=np.int64)

    abnormal_idx = np.where(pred_binary == 1)[0]

    if len(abnormal_idx) > 0:
        probs2 = stage2_model.predict(x_norm[abnormal_idx], verbose=0, batch_size=256)
        conf2 = np.max(probs2, axis=1)
        local_pred = np.argmax(probs2, axis=1).astype(np.int64)

        for j, idx in enumerate(abnormal_idx):
            if conf2[j] < stage2_confidence_threshold:
                final_pred[idx] = 4  # Q / Unknown
            else:
                # local 0/1/2 => raw 1/2/3
                final_pred[idx] = int(local_pred[j]) + 1

    # Evaluate only known labels N/S/V/F, excluding original Q=4
    known_mask = np.isin(y_raw, [0, 1, 2, 3])

    y_known = y_raw[known_mask].astype(np.int64)
    pred_known = final_pred[known_mask].astype(np.int64)

    labels = [0, 1, 2, 3, 4]
    names = ["Normal", "S", "V", "F", "Q/Unknown"]

    cm_known = confusion_matrix(y_known, pred_known, labels=labels)

    report_known = classification_report(
        y_known,
        pred_known,
        labels=labels,
        target_names=names,
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    svf_mask = np.isin(y_raw, [1, 2, 3])
    total_svf = int(np.sum(svf_mask))

    if total_svf:
        svf_pred = final_pred[svf_mask]
        unknown_on_svf = int(np.sum(svf_pred == 4))
        subtype_unknown_rate = unknown_on_svf / total_svf
        subtype_coverage = 1.0 - subtype_unknown_rate
    else:
        subtype_unknown_rate = 0.0
        subtype_coverage = 0.0

    return TwoStagePipelineResult(
        stage1_threshold=float(stage1_threshold),
        stage2_confidence_threshold=float(stage2_confidence_threshold),
        total_samples=int(len(y_raw)),
        normal_samples=int(np.sum(y_raw == 0)),
        abnormal_svf_samples=total_svf,
        q_unknown_samples=int(np.sum(y_raw == 4)),

        stage1_binary_accuracy=float(stage1_binary_accuracy),
        stage1_abnormal_precision=float(stage1_abnormal_precision),
        stage1_abnormal_recall=float(stage1_abnormal_recall),
        stage1_abnormal_f1=float(stage1_abnormal_f1),

        final_5class_accuracy_known_labels=float(accuracy_score(y_known, pred_known)),
        final_macro_f1_known_labels=float(f1_score(y_known, pred_known, labels=labels, average="macro", zero_division=0)),
        final_weighted_f1_known_labels=float(f1_score(y_known, pred_known, labels=labels, average="weighted", zero_division=0)),
        final_confusion_matrix_known_labels=cm_known.astype(int).tolist(),
        final_classification_report_known_labels=report_known,

        subtype_coverage_on_true_svf=float(subtype_coverage),
        subtype_unknown_rate_on_true_svf=float(subtype_unknown_rate),
    )


def save_outputs(
    reports_dir: Path,
    stage1: Stage1Result,
    stage2: Stage2Result,
    stage2_rej: Stage2RejectionResult,
    pipeline: TwoStagePipelineResult,
):
    reports_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = reports_dir / f"mitbih_v2_final_evaluation_{stamp}.json"
    csv_path = reports_dir / f"mitbih_v2_final_evaluation_{stamp}.csv"

    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage1_v2": asdict(stage1),
        "stage2_v2_known": asdict(stage2),
        "stage2_v2_with_rejection": asdict(stage2_rej),
        "two_stage_pipeline": asdict(pipeline),
    }

    json_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["section", "metric", "value"])

        writer.writerow(["Stage1", "threshold", stage1.threshold])
        writer.writerow(["Stage1", "accuracy", stage1.accuracy])
        writer.writerow(["Stage1", "precision_abnormal", stage1.precision_abnormal])
        writer.writerow(["Stage1", "recall_abnormal", stage1.recall_abnormal])
        writer.writerow(["Stage1", "f1_abnormal", stage1.f1_abnormal])
        writer.writerow(["Stage1", "specificity_normal", stage1.specificity_normal])
        writer.writerow(["Stage1", "roc_auc", stage1.roc_auc])
        writer.writerow(["Stage1", "tp", stage1.tp])
        writer.writerow(["Stage1", "tn", stage1.tn])
        writer.writerow(["Stage1", "fp", stage1.fp])
        writer.writerow(["Stage1", "fn", stage1.fn])

        writer.writerow(["Stage2 known", "accuracy", stage2.accuracy])
        writer.writerow(["Stage2 known", "macro_precision", stage2.macro_precision])
        writer.writerow(["Stage2 known", "macro_recall", stage2.macro_recall])
        writer.writerow(["Stage2 known", "macro_f1", stage2.macro_f1])
        writer.writerow(["Stage2 known", "weighted_f1", stage2.weighted_f1])

        writer.writerow(["Stage2 rejection", "confidence_threshold", stage2_rej.confidence_threshold])
        writer.writerow(["Stage2 rejection", "coverage", stage2_rej.coverage])
        writer.writerow(["Stage2 rejection", "unknown_rate", stage2_rej.unknown_rate])
        writer.writerow(["Stage2 rejection", "covered_accuracy", stage2_rej.covered_accuracy])
        writer.writerow(["Stage2 rejection", "covered_macro_f1", stage2_rej.covered_macro_f1])
        writer.writerow(["Stage2 rejection", "covered_weighted_f1", stage2_rej.covered_weighted_f1])

        writer.writerow(["Two-stage pipeline", "stage1_threshold", pipeline.stage1_threshold])
        writer.writerow(["Two-stage pipeline", "stage2_confidence_threshold", pipeline.stage2_confidence_threshold])
        writer.writerow(["Two-stage pipeline", "stage1_binary_accuracy", pipeline.stage1_binary_accuracy])
        writer.writerow(["Two-stage pipeline", "stage1_abnormal_precision", pipeline.stage1_abnormal_precision])
        writer.writerow(["Two-stage pipeline", "stage1_abnormal_recall", pipeline.stage1_abnormal_recall])
        writer.writerow(["Two-stage pipeline", "stage1_abnormal_f1", pipeline.stage1_abnormal_f1])
        writer.writerow(["Two-stage pipeline", "final_5class_accuracy_known_labels", pipeline.final_5class_accuracy_known_labels])
        writer.writerow(["Two-stage pipeline", "final_macro_f1_known_labels", pipeline.final_macro_f1_known_labels])
        writer.writerow(["Two-stage pipeline", "final_weighted_f1_known_labels", pipeline.final_weighted_f1_known_labels])
        writer.writerow(["Two-stage pipeline", "subtype_coverage_on_true_svf", pipeline.subtype_coverage_on_true_svf])
        writer.writerow(["Two-stage pipeline", "subtype_unknown_rate_on_true_svf", pipeline.subtype_unknown_rate_on_true_svf])

    print("\nSaved:")
    print(json_path)
    print(csv_path)


def main():
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "Dataset"
    models_dir = project_root / "Models"
    reports_dir = project_root / "Reports"

    stage1_path = models_dir / "mitbih_stage1_binary_v2.keras"
    stage1_meta_path = models_dir / "mitbih_stage1_binary_v2_meta.json"

    stage2_path = models_dir / "mitbih_stage2_subtypes_svf_v2.keras"
    stage2_meta_path = models_dir / "mitbih_stage2_subtypes_svf_v2_meta.json"

    stage1_threshold = load_threshold_from_meta(stage1_meta_path, default=0.76)
    stage2_conf_threshold = load_conf_threshold_from_meta(stage2_meta_path, default=0.90)

    print("Stage 1 threshold:", stage1_threshold)
    print("Stage 2 confidence threshold:", stage2_conf_threshold)

    print("Loading test data...")
    x_test, y_raw = load_test_full(dataset_dir)

    print("Loading models...")
    stage1_model = load_model(stage1_path)
    stage2_model = load_model(stage2_path)

    print("\nEvaluating Stage 1 v2...")
    stage1_result = evaluate_stage1(
        stage1_model,
        x_test,
        y_raw,
        threshold=stage1_threshold,
    )

    print("\nEvaluating Stage 2 v2 known S/V/F...")
    stage2_result = evaluate_stage2_known(
        stage2_model,
        x_test,
        y_raw,
    )

    print("\nEvaluating Stage 2 v2 with confidence rejection...")
    stage2_rejection_result = evaluate_stage2_rejection(
        stage2_model,
        x_test,
        y_raw,
        confidence_threshold=stage2_conf_threshold,
    )

    print("\nEvaluating full two-stage pipeline...")
    pipeline_result = evaluate_two_stage_pipeline(
        stage1_model,
        stage2_model,
        x_test,
        y_raw,
        stage1_threshold=stage1_threshold,
        stage2_confidence_threshold=stage2_conf_threshold,
    )

    print("\n================ SUMMARY ================")

    print(
        f"Stage 1 v2: "
        f"ACC={stage1_result.accuracy:.4f}, "
        f"PREC_ABN={stage1_result.precision_abnormal:.4f}, "
        f"REC_ABN={stage1_result.recall_abnormal:.4f}, "
        f"F1_ABN={stage1_result.f1_abnormal:.4f}, "
        f"AUC={stage1_result.roc_auc:.4f}"
    )

    print(
        f"Stage 2 v2 known: "
        f"ACC={stage2_result.accuracy:.4f}, "
        f"MACRO_F1={stage2_result.macro_f1:.4f}, "
        f"WEIGHTED_F1={stage2_result.weighted_f1:.4f}"
    )

    print(
        f"Stage 2 v2 rejection: "
        f"CONF_THR={stage2_rejection_result.confidence_threshold:.2f}, "
        f"COVERAGE={stage2_rejection_result.coverage:.4f}, "
        f"UNKNOWN={stage2_rejection_result.unknown_rate:.4f}, "
        f"COVERED_ACC={stage2_rejection_result.covered_accuracy:.4f}, "
        f"COVERED_MACRO_F1={stage2_rejection_result.covered_macro_f1:.4f}"
    )

    print(
        f"Two-stage pipeline: "
        f"FINAL_5CLASS_ACC={pipeline_result.final_5class_accuracy_known_labels:.4f}, "
        f"FINAL_MACRO_F1={pipeline_result.final_macro_f1_known_labels:.4f}, "
        f"FINAL_WEIGHTED_F1={pipeline_result.final_weighted_f1_known_labels:.4f}, "
        f"SUBTYPE_COVERAGE={pipeline_result.subtype_coverage_on_true_svf:.4f}"
    )

    save_outputs(
        reports_dir,
        stage1_result,
        stage2_result,
        stage2_rejection_result,
        pipeline_result,
    )


if __name__ == "__main__":
    main()