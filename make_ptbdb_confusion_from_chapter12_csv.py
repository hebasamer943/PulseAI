from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = Path(
    r"C:\Users\hebas\OneDrive\Desktop\Project ECG\Reports\chapter12_ptbdb_confusion_matrix.csv"
)

OUT_DIR = Path(
    r"C:\Users\hebas\OneDrive\Desktop\Project ECG\Reports\chapter12_figures"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_IMAGE = OUT_DIR / "Figure_12_1_PTBDB_Final_Live_Pipeline_Confusion_Matrix.png"


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
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path.resolve())


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"File not found: {CSV_PATH}")

    final_row = None

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            model_name = row["model"].strip().lower()

            if "final live pipeline" in model_name:
                final_row = row
                break

    if final_row is None:
        raise ValueError("Final Live Pipeline row was not found in the CSV file.")

    tn = int(final_row["tn"])
    fp = int(final_row["fp"])
    fn = int(final_row["fn"])
    tp = int(final_row["tp"])

    cm = [
        [tn, fp],
        [fn, tp],
    ]

    print("Using confusion matrix:", cm)

    plot_confusion_matrix(
        cm,
        labels=["Normal", "Abnormal"],
        title="PTBDB Final Live Pipeline Confusion Matrix",
        out_path=OUT_IMAGE,
    )


if __name__ == "__main__":
    main()