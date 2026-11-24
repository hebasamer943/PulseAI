from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "AI" / "models_experiments" / "cache" / "mitbih_5class_ds1train_ds1val_ds2_187.npz"
MODELS_DIR = PROJECT_ROOT / "Models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "mitbih_stage2_subtypes_svf.keras"
META_PATH = MODELS_DIR / "mitbih_stage2_subtypes_svf_meta.json"
BEST_WEIGHTS = MODELS_DIR / "_tmp_mitbih_stage2_subtypes_svf_best.weights.h5"

DEFAULT_EPOCHS = 45
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-4
PATIENCE = 9

MIT_TO_LOCAL = {1: 0, 2: 1, 3: 2}
LOCAL_TO_LABEL = {0: "S", 1: "V", 2: "F"}
DISPLAY_NAMES = {
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion beat",
    "Q": "Unknown / low confidence",
}


def z_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return (x - mu) / sd


def load_data():
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing cache file: {CACHE_PATH}")

    d = np.load(CACHE_PATH)

    def filter_abnormal(x: np.ndarray, y: np.ndarray):
        mask = np.isin(y, [1, 2, 3])
        x = z_norm(x[mask])[..., None].astype(np.float32)
        y = y[mask].astype(np.int64)
        y = np.asarray([MIT_TO_LOCAL[int(v)] for v in y], dtype=np.int64)
        return x, y

    x_train, y_train = filter_abnormal(d["X_train"], d["y_train"])
    x_val, y_val = filter_abnormal(d["X_val"], d["y_val"])
    x_test, y_test = filter_abnormal(d["X_test"], d["y_test"])
    return x_train, y_train, x_val, y_val, x_test, y_test


def oversample_subtypes(x: np.ndarray, y: np.ndarray):
    rng = np.random.default_rng(SEED)

    idx_s = np.where(y == 0)[0]
    idx_v = np.where(y == 1)[0]
    idx_f = np.where(y == 2)[0]

    max_count = max(len(idx_s), len(idx_v), len(idx_f))
    target_s = max(len(idx_s), int(max_count * 0.90))
    target_f = max(len(idx_f), int(max_count * 0.70))

    extra_s = (
        rng.choice(idx_s, size=max(0, target_s - len(idx_s)), replace=True)
        if len(idx_s)
        else np.array([], dtype=np.int64)
    )
    extra_f = (
        rng.choice(idx_f, size=max(0, target_f - len(idx_f)), replace=True)
        if len(idx_f)
        else np.array([], dtype=np.int64)
    )

    idx = np.concatenate([idx_s, idx_v, idx_f, extra_s, extra_f])
    rng.shuffle(idx)
    return x[idx], y[idx]


def focal_loss(class_weights=(1.5, 1.0, 2.6), gamma=1.4):
    cw = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_one = tf.one_hot(y_true, depth=3)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = -tf.reduce_sum(y_one * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_one * y_pred, axis=-1)
        alpha = tf.gather(cw, y_true)
        return tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * ce)

    return loss


def se_block(x, ratio=8):
    c = int(x.shape[-1])
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(c // ratio, 8), activation="relu")(s)
    s = layers.Dense(c, activation="sigmoid")(s)
    s = layers.Reshape((1, c))(s)
    return layers.Multiply()([x, s])


def res_block(x, filters, kernel_size=7, drop=0.18):
    shortcut = x
    if int(x.shape[-1]) != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Conv1D(filters, kernel_size, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.SpatialDropout1D(drop)(y)
    y = layers.Conv1D(filters, kernel_size, padding="same")(y)
    y = layers.BatchNormalization()(y)
    y = se_block(y)
    y = layers.Add()([shortcut, y])
    y = layers.Activation("relu")(y)
    return y


def build_model(lr: float):
    inp = keras.Input(shape=(187, 1))
    x = layers.Conv1D(32, 7, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = res_block(x, 32, 7, 0.10)
    x = res_block(x, 64, 5, 0.15)
    x = layers.MaxPool1D(2)(x)
    x = res_block(x, 96, 5, 0.20)
    x = res_block(x, 128, 3, 0.25)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=focal_loss(class_weights=(1.5, 1.0, 2.6), gamma=1.4),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def augment_batch(x, y):
    batch = tf.shape(x)[0]

    x = x + tf.random.normal(tf.shape(x), stddev=0.02, dtype=x.dtype)
    x = x * tf.random.uniform((batch, 1, 1), 0.95, 1.05, dtype=x.dtype)

    shift = tf.random.uniform((batch,), minval=-4, maxval=5, dtype=tf.int32)
    x = tf.map_fn(
        lambda z: tf.roll(z[0], shift=z[1], axis=0),
        (x, shift),
        fn_output_signature=tf.float32,
    )

    t = tf.linspace(0.0, 1.0, tf.shape(x)[1])
    t = tf.reshape(t, (1, -1, 1))
    phase = tf.random.uniform((batch, 1, 1), 0.0, np.pi, dtype=x.dtype)
    amp = tf.random.uniform((batch, 1, 1), 0.0, 0.06, dtype=x.dtype)
    wander = amp * tf.sin((2.0 * np.pi * 1.1 * t) + phase)
    x = x + wander

    return x, y


def make_ds(x: np.ndarray, y: np.ndarray, batch_size: int, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(y), 20000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if training:
        ds = ds.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


class BestMacroF1Callback(keras.callbacks.Callback):
    def __init__(self, x_val: np.ndarray, y_val: np.ndarray):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.best = -1.0

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.x_val, verbose=0)
        pred = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)
        score = float(f1_score(self.y_val, pred, average="macro", zero_division=0))
        print(f"\n[val] epoch={epoch + 1} macro_f1={score:.4f}")
        if score > self.best:
            self.best = score
            self.model.save_weights(BEST_WEIGHTS)


def predict_probs(model: keras.Model, x: np.ndarray) -> np.ndarray:
    return model.predict(x, verbose=0, batch_size=256)


def search_rejection_threshold(y_true: np.ndarray, probs: np.ndarray):
    conf = np.max(probs, axis=1)
    pred = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)

    best_thr = 0.60
    best_score = -1.0

    for thr in np.arange(0.45, 0.86, 0.02):
        covered = conf >= thr
        coverage = float(np.mean(covered))
        if coverage < 0.55:
            continue

        macro_f1_cov = float(f1_score(y_true[covered], pred[covered], average="macro", zero_division=0))
        unknown_rate = 1.0 - coverage
        score = macro_f1_cov - (0.15 * unknown_rate)

        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr, best_score


def evaluate_known(name: str, y_true: np.ndarray, probs: np.ndarray):
    pred = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)
    print(f"\n=== {name} subtype metrics (S/V/F) ===")
    print("Accuracy:", accuracy_score(y_true, pred))
    print("Macro F1:", f1_score(y_true, pred, average="macro", zero_division=0))
    print("Weighted F1:", f1_score(y_true, pred, average="weighted", zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true, pred, labels=[0, 1, 2]))
    print(
        "\nClassification Report:\n",
        classification_report(
            y_true,
            pred,
            labels=[0, 1, 2],
            target_names=["S", "V", "F"],
            digits=4,
            zero_division=0,
        ),
    )


def evaluate_with_rejection(name: str, y_true: np.ndarray, probs: np.ndarray, threshold: float):
    conf = np.max(probs, axis=1)
    pred = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)
    covered = conf >= threshold
    coverage = float(np.mean(covered))

    print(f"\n=== {name} subtype with confidence rejection @ {threshold:.2f} ===")
    print("Coverage:", int(np.sum(covered)), "/", len(y_true))
    print("Unknown rate:", 1.0 - coverage)

    if not np.any(covered):
        print("No covered samples.")
        return

    y_cov = y_true[covered]
    pred_cov = pred[covered]
    print("Covered Accuracy:", accuracy_score(y_cov, pred_cov))
    print("Covered Macro F1:", f1_score(y_cov, pred_cov, average="macro", zero_division=0))
    print("Covered Confusion Matrix:\n", confusion_matrix(y_cov, pred_cov, labels=[0, 1, 2]))
    print(
        "\nCovered Classification Report:\n",
        classification_report(
            y_cov,
            pred_cov,
            labels=[0, 1, 2],
            target_names=["S", "V", "F"],
            digits=4,
            zero_division=0,
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train MIT-BIH stage 2 subtype model: S/V/F on abnormal beats only.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    return parser.parse_args()


def main():
    args = parse_args()

    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    x_train, y_train = oversample_subtypes(x_train, y_train)

    print("Train:", x_train.shape, {"S": int((y_train == 0).sum()), "V": int((y_train == 1).sum()), "F": int((y_train == 2).sum())})
    print("Val  :", x_val.shape, {"S": int((y_val == 0).sum()), "V": int((y_val == 1).sum()), "F": int((y_val == 2).sum())})
    print("Test :", x_test.shape, {"S": int((y_test == 0).sum()), "V": int((y_test == 1).sum()), "F": int((y_test == 2).sum())})

    model = build_model(lr=float(args.lr))
    model.fit(
        make_ds(x_train, y_train, batch_size=int(args.batch_size), training=True),
        validation_data=make_ds(x_val, y_val, batch_size=int(args.batch_size), training=False),
        epochs=int(args.epochs),
        callbacks=[
            BestMacroF1Callback(x_val, y_val),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=False),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ],
        verbose=1,
    )

    if BEST_WEIGHTS.exists():
        model.load_weights(BEST_WEIGHTS)

    val_probs = predict_probs(model, x_val)
    test_probs = predict_probs(model, x_test)

    evaluate_known("VAL", y_val, val_probs)
    evaluate_known("TEST", y_test, test_probs)

    conf_thr, conf_score = search_rejection_threshold(y_val, val_probs)
    evaluate_with_rejection("VAL", y_val, val_probs, conf_thr)
    evaluate_with_rejection("TEST", y_test, test_probs, conf_thr)

    model.save(MODEL_PATH)
    meta = {
        "task": "stage2_subtypes_svf",
        "train_class_names": ["S", "V", "F"],
        "display_names": DISPLAY_NAMES,
        "original_mitbih_ids": {"S": 1, "V": 2, "F": 3},
        "recommended_confidence_threshold": float(conf_thr),
        "selection_score": float(conf_score),
        "low_confidence_label": "Q",
        "input_length": 187,
        "normalization": "per-beat z-score",
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved meta  -> {META_PATH}")
    print(f"Recommended subtype confidence threshold -> {conf_thr:.2f}")


if __name__ == "__main__":
    main()
