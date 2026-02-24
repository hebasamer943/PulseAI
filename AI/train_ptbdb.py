from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "Dataset"
MODELS = BASE / "Models"
MODELS.mkdir(exist_ok=True)

NORMAL = DATA / "ptbdb_normal.csv"
ABNORMAL = DATA / "ptbdb_abnormal.csv"

def load_csv(path: Path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.int64)
    return X, y

def main():
    print("=== Training PTBDB (Normal vs Abnormal) ===")
    print("Dataset folder:", DATA)

    if not NORMAL.exists():
        raise FileNotFoundError(f"Missing file: {NORMAL}")
    if not ABNORMAL.exists():
        raise FileNotFoundError(f"Missing file: {ABNORMAL}")

    Xn, yn = load_csv(NORMAL)
    Xa, ya = load_csv(ABNORMAL)

    print("Normal:", Xn.shape, "labels:", np.unique(yn))
    print("Abnormal:", Xa.shape, "labels:", np.unique(ya))

    X = np.vstack([Xn, Xa])
    y = np.concatenate([yn, ya])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=4))

    out = MOD
