from dataclasses import dataclass
import numpy as np


@dataclass
class QualityResult:
    score: float   # 0..1
    label: str
    hint: str


def evaluate_quality(beat: np.ndarray) -> QualityResult:
    
    x = np.asarray(beat, dtype=np.float32).reshape(-1)
    n = x.size
    if n < 50:
        return QualityResult(0.0, "POOR", "No signal")

    # Remove baseline
    x = x - np.median(x)

    # Amplitude (very small -> poor)
    raw_peak = float(np.max(np.abs(x)) + 1e-9)
    if raw_peak < 0.02:
        return QualityResult(0.15, "POOR", "Very low amplitude (check electrodes / gain)")

    # Normalize
    xn = x / raw_peak

    # Smooth and compute residual noise 
    k = 9
    kernel = np.ones(k, dtype=np.float32) / k
    smooth = np.convolve(xn, kernel, mode="same")
    resid = xn - smooth

    sig_rms = float(np.sqrt(np.mean(smooth * smooth)) + 1e-9)
    noise_rms = float(np.sqrt(np.mean(resid * resid)) + 1e-9)
    noise_ratio = noise_rms / (sig_rms + 1e-9)  # higher = noisier

    # Baseline drift 
    w = max(12, n // 8)
    drift = float(abs(np.mean(xn[:w]) - np.mean(xn[-w:])))

    # Clipping
    clip = float(np.mean((np.abs(np.abs(xn) - 1.0) < 0.01)))

    # Multi-peak (soft penalty)
    a = np.abs(smooth)
    thr = 0.65
    peaks = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] > thr))[0]
    peak_count = int(peaks.size)

    # Score
    score = 1.0
    # noise penalty softer
    score -= min(0.55, noise_ratio * 0.80)
    # drift penalty
    score -= min(0.45, drift * 0.90)

    if clip > 0.03:
        score -= 0.25

    if peak_count == 0:
        score -= 0.20
    elif peak_count >= 4:
        score -= 0.20
    elif peak_count == 3:
        score -= 0.10

    score = float(np.clip(score, 0.0, 1.0))

    if score < 0.35:
        return QualityResult(score, "POOR", "Poor contact / motion noise (try staying still)")
    if score < 0.60:
        return QualityResult(score, "FAIR", "Usable but noisy")
    if score < 0.80:
        return QualityResult(score, "GOOD", "Good signal")
    return QualityResult(score, "EXCELLENT", "Very clean signal")