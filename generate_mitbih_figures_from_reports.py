from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


REPORTS_DIR = Path(r"C:\Users\hebas\OneDrive\Desktop\Project ECG\Reports")

STAGE1_REPORT = REPORTS_DIR / "mitbih_stage1_v2_report_20260501_031450.json"
STAGE2_REPORT = REPORTS_DIR / "mitbih_stage2_v2_report_20260501_085947.json"

OUT_DIR = REPORTS_DIR / "chapter12_figures"
OUT_DIR.mkdir(exist_ok=True)

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_loss_curve(history, title, out_path):
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    plt.figure(figsize=(7, 5))
    plt.plot(loss, label="Training Loss")

    if val_loss:
        plt.plot(val_loss, label="Validation Loss")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)


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


def save_summary_text(text, out_path):
    out_path.write_text(text, encoding="utf-8")
    print("Saved:", out_path)


def main():
    stage1 = load_json(STAGE1_REPORT)
    stage2 = load_json(STAGE2_REPORT)

    # =========================
    # Stage 1 Loss Curve
    # =========================
    plot_loss_curve(
        stage1["history"],
        "MIT-BIH Stage 1 Training and Validation Loss",
        OUT_DIR / "Figure_12_4_MITBIH_Stage1_Loss_Curve.png",
    )

    # =========================
    # Stage 1 Confusion Matrix
    # =========================
    stage1_cm = stage1["test_result"]["confusion_matrix"]

    plot_confusion_matrix(
        stage1_cm,
        labels=["Normal-like", "Abnormal"],
        title="MIT-BIH Stage 1 Confusion Matrix",
        out_path=OUT_DIR / "Figure_12_2_MITBIH_Stage1_Confusion_Matrix.png",
    )

    stage1_text = (
        "MIT-BIH Stage 1 Test Result\n"
        f"Accuracy: {stage1['test_result']['accuracy']:.4f}\n"
        f"Precision abnormal: {stage1['test_result']['precision_abnormal']:.4f}\n"
        f"Recall abnormal: {stage1['test_result']['recall_abnormal']:.4f}\n"
        f"F1 abnormal: {stage1['test_result']['f1_abnormal']:.4f}\n"
        f"ROC-AUC: {stage1['test_result']['roc_auc']:.4f}\n"
        f"Threshold: {stage1['test_result']['threshold']:.2f}\n"
        f"Confusion Matrix: {stage1_cm}\n"
    )

    save_summary_text(stage1_text, OUT_DIR / "MITBIH_Stage1_Summary.txt")

    # =========================
    # Stage 2 Loss Curve
    # =========================
    plot_loss_curve(
        stage2["history"],
        "MIT-BIH Stage 2 Training and Validation Loss",
        OUT_DIR / "Figure_12_5_MITBIH_Stage2_Loss_Curve.png",
    )

    # =========================
    # Stage 2 Known Confusion Matrix
    # =========================
    stage2_cm = stage2["test_known_result"]["confusion_matrix"]

    plot_confusion_matrix(
        stage2_cm,
        labels=["S", "V", "F"],
        title="MIT-BIH Stage 2 S/V/F Confusion Matrix",
        out_path=OUT_DIR / "Figure_12_3_MITBIH_Stage2_SVF_Confusion_Matrix.png",
    )

    # =========================
    # Stage 2 Covered Confusion Matrix
    # Optional, but useful
    # =========================
    stage2_cov_cm = stage2["test_rejection_result"]["covered_confusion_matrix"]

    plot_confusion_matrix(
        stage2_cov_cm,
        labels=["S", "V", "F"],
        title="MIT-BIH Stage 2 Covered Predictions Confusion Matrix",
        out_path=OUT_DIR / "Figure_12_3B_MITBIH_Stage2_Covered_Confusion_Matrix.png",
    )

    stage2_text = (
        "MIT-BIH Stage 2 Test Result\n"
        f"Accuracy: {stage2['test_known_result']['accuracy']:.4f}\n"
        f"Macro Precision: {stage2['test_known_result']['macro_precision']:.4f}\n"
        f"Macro Recall: {stage2['test_known_result']['macro_recall']:.4f}\n"
        f"Macro F1: {stage2['test_known_result']['macro_f1']:.4f}\n"
        f"Weighted F1: {stage2['test_known_result']['weighted_f1']:.4f}\n"
        f"Confusion Matrix: {stage2_cm}\n\n"
        "With confidence rejection:\n"
        f"Confidence Threshold: {stage2['test_rejection_result']['confidence_threshold']:.2f}\n"
        f"Coverage: {stage2['test_rejection_result']['coverage']:.4f}\n"
        f"Unknown Rate: {stage2['test_rejection_result']['unknown_rate']:.4f}\n"
        f"Covered Accuracy: {stage2['test_rejection_result']['covered_accuracy']:.4f}\n"
        f"Covered Macro F1: {stage2['test_rejection_result']['covered_macro_f1']:.4f}\n"
        f"Covered Confusion Matrix: {stage2_cov_cm}\n"
    )

    save_summary_text(stage2_text, OUT_DIR / "MITBIH_Stage2_Summary.txt")

    print("\nDone. Check the folder:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()