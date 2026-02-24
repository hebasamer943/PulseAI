from pathlib import Path
import numpy as np
import tensorflow as tf

# جذر المشروع = Project ECG
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Models" / "ptbdb_resnet1d_fix.keras"

THRESHOLD = 0.60
_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_ecg(values_187):
    x = np.asarray(values_187, dtype=np.float32).reshape(1, 187, 1)
    prob = float(_model.predict(x, verbose=0).reshape(-1)[0])
    label = "Abnormal" if prob >= THRESHOLD else "Normal"
    return label, prob
