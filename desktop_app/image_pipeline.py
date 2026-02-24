from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class ExtractResult:
    beat_187: np.ndarray
    ok: bool
    msg: str

def _resample_to_187(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < 2:
        return np.zeros((187,), np.float32)
    x_old = np.linspace(0.0, 1.0, num=y.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=187, dtype=np.float32)
    return np.interp(x_new, x_old, y).astype(np.float32)

def extract_ecg_from_image(image_path: str) -> ExtractResult:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return ExtractResult(np.zeros(187, np.float32), False, "Cannot read image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    h, w = bw.shape
    ys = np.full((w,), np.nan, dtype=np.float32)
    for xi in range(w):
        idx = np.where(bw[:, xi] > 0)[0]
        if idx.size:
            ys[xi] = float(idx.min())

    mask = np.isfinite(ys)
    if mask.sum() < max(20, w * 0.05):
        return ExtractResult(np.zeros(187, np.float32), False, "Trace not detected (try clearer image / crop ECG area)")

    x = np.arange(w, dtype=np.float32)
    ys = np.interp(x, x[mask], ys[mask]).astype(np.float32)

    amp = -(ys - np.median(ys))
    amp = amp / (np.max(np.abs(amp)) + 1e-9)

    return ExtractResult(_resample_to_187(amp), True, "OK")
