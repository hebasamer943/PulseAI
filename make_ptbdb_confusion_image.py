from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


RESULT_JSON = Path(
    r"C:\Users\hebas\OneDrive\Desktop\Project ECG\Reports\chapter12_figures\PTBDB_Final_Live_Pipeline_Result.json"
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
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path.resolve())


def main():
    if not RESULT_JSON.exists():
        raise FileNotFoundError(f"File not found: {RESULT_JSON.resolve()}")

    data = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    cm = data["confusion_matrix"]

    plot_confusion_matrix(
        cm,
        labels=["Normal", "Abnormal"],
        title="PTBDB Final Live Pipeline Confusion Matrix",
        out_path=OUT_IMAGE,
    )


if __name__ == "__main__":
    main()