from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parents[2]   # Project ECG
DATA = BASE / "Dataset"
MODELS = BASE / "Models"
MODELS.mkdir(exist_ok=True)

NORMAL = DATA / "ptbdb_normal.csv"
ABNORMAL = DATA / "ptbdb_abnormal.csv"

def load_ptbdb():
    dn = pd.read_csv(NORMAL, header=None)
    da = pd.read_csv(ABNORMAL, header=None)
    Xn, yn = dn.iloc[:, :-1].to_numpy(np.float32), dn.iloc[:, -1].to_numpy(np.int64)
    Xa, ya = da.iloc[:, :-1].to_numpy(np.float32), da.iloc[:, -1].to_numpy(np.int64)
    X = np.vstack([Xn, Xa])
    y = np.concatenate([yn, ya])
    return X, y

def main():
    print("=== RandomForest PTBDB ===")
    X, y = load_ptbdb()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print("Accuracy:", round(acc, 4), "F1(abnormal):", round(f1, 4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=4))

    out = MODELS / "ptbdb_rf.joblib"
    joblib.dump(clf, out)
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
