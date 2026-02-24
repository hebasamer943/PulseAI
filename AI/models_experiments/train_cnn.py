from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE = Path(__file__).resolve().parents[2]
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
    print("=== 1D-CNN PTBDB ===")
    X, y = load_ptbdb()

    # reshape to (N, 187, 1)
    X = X.reshape(-1, 187, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = keras.Sequential([
        layers.Input(shape=(187, 1)),
        layers.Conv1D(32, 7, padding="same"), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPool1D(2),

        layers.Conv1D(64, 5, padding="same"), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPool1D(2),

        layers.Conv1D(128, 3, padding="same"), layers.BatchNormalization(), layers.ReLU(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc"), keras.metrics.AUC(name="auc")]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2),
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    # evaluate
    prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print("Accuracy:", round(acc, 4), "F1(abnormal):", round(f1, 4))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=4))

    out = MODELS / "ptbdb_cnn.keras"
    model.save(out)
    print("\nSaved:", out)

if __name__ == "__main__":
    main()
