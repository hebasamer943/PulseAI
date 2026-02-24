from pathlib import Path
import numpy as np
import joblib

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "Models" / "ptbdb_lr.joblib"

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_beat(beat187: np.ndarray):
    """
    beat187: numpy array length 187
    returns: (pred_label 0/1, prob_abnormal)
    """
    m = load_model()
    x = beat187.astype(np.float32).reshape(1, -1)
    prob_abn = float(m.predict_proba(x)[0, 1])
    pred = int(prob_abn >= 0.5)
    return pred, prob_abn
