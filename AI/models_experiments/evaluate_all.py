from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import joblib
import tensorflow as tf

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "Dataset"
MODELS = BASE / "Models"

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

def eval_probs(name, y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, pos_label=1)
    cm = confusion_matrix(y_true, pred)
    return {
        "model": name,
        "threshold": float(threshold),
        "accuracy": float(acc),
        "f1_abnormal": float(f1),
        "TN": int(cm[0,0]), "FP": int(cm[0,1]),
        "FN": int(cm[1,0]), "TP": int(cm[1,1]),
    }
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model

        pos = np.arange(length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angles = pos * angle_rates

        pe = np.zeros((length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :])

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "d_model": self.d_model})
        return cfg

def main():
    X, y = load_ptbdb()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []

    # LightGBM / XGB (joblib)
    lgbm_path = MODELS / "ptbdb_lgbm.joblib"
    if lgbm_path.exists():
        m = joblib.load(lgbm_path)
        prob = m.predict_proba(X_test)[:, 1]
        results.append(eval_probs("LightGBM", y_test, prob, threshold=0.5))

    # RandomForest (joblib)
    rf_path = MODELS / "ptbdb_rf.joblib"
    if rf_path.exists():
        m = joblib.load(rf_path)
        prob = m.predict_proba(X_test)[:, 1]
        results.append(eval_probs("RandomForest", y_test, prob, threshold=0.5))

    # Transformer (keras)
    tf_path = MODELS / "ptbdb_transformer_fix.keras"
    if tf_path.exists():
        m = tf.keras.models.load_model(
      tf_path,
      compile=False,
      custom_objects={"PositionalEncoding": PositionalEncoding},
        )

        prob = m.predict(X_test.reshape(-1,187,1), verbose=0).reshape(-1)
        results.append(eval_probs("Transformer_fix", y_test, prob, threshold=0.5))

    # ResNet1D Fix (keras) — (هنا استخدمي threshold اللي طلع عندك أفضل: 0.60)
    resnet_path = MODELS / "ptbdb_resnet1d_fix.keras"
    if resnet_path.exists():
        m = tf.keras.models.load_model(resnet_path, compile=False)
        prob = m.predict(X_test.reshape(-1,187,1), verbose=0).reshape(-1)
        results.append(eval_probs("ResNet1D_fix", y_test, prob, threshold=0.60))

    # TCN (keras) — خليها 0.5 كبداية أو عدليها إذا عندك threshold أفضل
    tcn_path = MODELS / "ptbdb_tcn.keras"
    if tcn_path.exists():
        m = tf.keras.models.load_model(tcn_path, compile=False)
        prob = m.predict(X_test.reshape(-1,187,1), verbose=0).reshape(-1)
        results.append(eval_probs("TCN", y_test, prob, threshold=0.5))

    df = pd.DataFrame(results).sort_values(["accuracy","f1_abnormal"], ascending=False)
    print("\n=== FINAL COMPARISON ===")
    print(df.to_string(index=False))

    out = BASE / "Reports"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "final_model_comparison.csv", index=False)
    print("\nSaved:", out / "final_model_comparison.csv")

if __name__ == "__main__":
    main()
