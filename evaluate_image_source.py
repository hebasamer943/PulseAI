from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model

        pos = np.arange(length)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angles = pos * angle_rates

        pe = np.zeros((length, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.constant(pe[None, :, :])

    def call(self, x):
        return x + self.pe[:, : tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "d_model": self.d_model})
        return cfg


@dataclass
class EvalMetrics:
    n_images: int
    n_labeled: int
    n_extracted_ok: int
    n_extracted_fail: int
    extraction_success_rate: float
    n_classified_labeled: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    specificity: float | None
    f1: float | None
    tp: int
    tn: int
    fp: int
    fn: int


def normalize_beat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size > 187:
        x = x[:187]
    if x.size < 187:
        x = np.pad(x, (0, 187 - x.size))
    x = x - np.median(x)
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x.astype(np.float32)


def infer_label_from_path(path: Path) -> int | None:
    text = " ".join([path.name.lower(), str(path.parent).lower()])

    # Put abnormal checks first because "not_normal" may contain normal.
    abnormal_words = [
        "abnormal",
        "label_1",
        "class_1",
        "disease",
        "arrhythmia",
        "positive",
        "abn",
    ]
    normal_words = [
        "normal",
        "label_0",
        "class_0",
        "negative",
    ]

    if any(w in text for w in abnormal_words):
        return 1
    if any(w in text for w in normal_words):
        return 0
    return None


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


class ImagePredictionPipeline:
    """
    Mirrors the app's current image decision:
    1. Extract beat from image.
    2. Use main signal classifier as primary backend.
    3. Use image-specific model only as weak tiebreaker when it agrees.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)

        self.resnet_path = self.models_dir / "ptbdb_resnet1d_fix.keras"
        self.tcn_path = self.models_dir / "ptbdb_tcn.keras"
        self.trans_path = self.models_dir / "ptbdb_transformer_fix.keras"
        self.image_model_path = self.models_dir / "mitbih_binary_inception.keras"

        self.resnet = None
        self.tcn = None
        self.trans = None
        self.image_model = None

        self.w_resnet = 0.30
        self.w_tcn = 0.50
        self.w_trans = 0.20
        self.unc_low = 0.20
        self.unc_high = 0.85

    def load(self):
        if not self.resnet_path.exists():
            raise FileNotFoundError(f"Missing model: {self.resnet_path}")

        self.resnet = tf.keras.models.load_model(self.resnet_path, compile=False)

        if self.tcn_path.exists():
            try:
                self.tcn = tf.keras.models.load_model(self.tcn_path, compile=False)
            except Exception as e:
                print(f"WARNING: failed to load TCN, using ResNet fallback. Error: {e}")

        if self.trans_path.exists():
            try:
                self.trans = tf.keras.models.load_model(
                    self.trans_path,
                    compile=False,
                    custom_objects={"PositionalEncoding": PositionalEncoding},
                )
            except Exception as e:
                print(f"WARNING: failed to load Transformer. Error: {e}")

        if self.image_model_path.exists():
            try:
                self.image_model = tf.keras.models.load_model(self.image_model_path, compile=False)
            except Exception as e:
                print(f"WARNING: failed to load image model. Error: {e}")

    def _predict_one_model(self, model, beat: np.ndarray) -> float:
        x = normalize_beat(beat).reshape(1, 187, 1)
        return float(model.predict(x, verbose=0).reshape(-1)[0])

    def predict_signal_pipeline(self, beat: np.ndarray) -> float:
        if self.resnet is None:
            self.load()

        if self.tcn is None:
            return self._predict_one_model(self.resnet, beat)

        p_tcn = self._predict_one_model(self.tcn, beat)

        if p_tcn < self.unc_low or p_tcn > self.unc_high:
            return float(p_tcn)

        weighted = [(self.w_tcn, p_tcn)]

        p_res = self._predict_one_model(self.resnet, beat)
        weighted.append((self.w_resnet, p_res))

        if self.trans is not None:
            p_tr = self._predict_one_model(self.trans, beat)
            weighted.append((self.w_trans, p_tr))

        total_w = sum(w for w, _ in weighted)
        return float(sum(w * p for w, p in weighted) / max(total_w, 1e-9))

    def predict_image_probability(self, beat: np.ndarray) -> float:
        x = normalize_beat(beat)

        p_signal = self.predict_signal_pipeline(x)
        p_final = p_signal

        if self.image_model is not None:
            p_image = self._predict_one_model(self.image_model, x)

            same_direction = (
                (p_signal >= 0.55 and p_image >= 0.55)
                or
                (p_signal <= 0.45 and p_image <= 0.45)
            )

            if same_direction:
                p_final = (0.90 * p_signal) + (0.10 * p_image)

        return float(np.clip(p_final, 0.0, 1.0))


def compute_metrics(rows: list[dict], threshold: float) -> EvalMetrics:
    n_images = len(rows)
    labeled = [r for r in rows if r["true_label"] in (0, 1)]
    extracted_ok = [r for r in rows if r["extract_ok"]]
    extracted_fail = [r for r in rows if not r["extract_ok"]]
    classified_labeled = [
        r for r in rows
        if r["extract_ok"] and r["true_label"] in (0, 1) and r["prob_abnormal"] != ""
    ]

    tp = tn = fp = fn = 0
    for r in classified_labeled:
        y = int(r["true_label"])
        p = float(r["prob_abnormal"])
        pred = 1 if p >= threshold else 0

        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1

    n_cls = len(classified_labeled)

    if n_cls == 0:
        acc = prec = rec = spec = f1 = None
    else:
        acc = safe_div(tp + tn, n_cls)
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)

        if prec is None or rec is None or (prec + rec) == 0:
            f1 = None
        else:
            f1 = 2.0 * prec * rec / (prec + rec)

    return EvalMetrics(
        n_images=n_images,
        n_labeled=len(labeled),
        n_extracted_ok=len(extracted_ok),
        n_extracted_fail=len(extracted_fail),
        extraction_success_rate=0.0 if n_images == 0 else len(extracted_ok) / n_images,
        n_classified_labeled=n_cls,
        accuracy=acc,
        precision=prec,
        recall=rec,
        specificity=spec,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ECG image source end-to-end.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    images_dir = args.images_dir.resolve()
    models_dir = project_root / "Models"

    sys.path.insert(0, str(project_root))
    from desktop_app.image_pipeline import extract_ecg_from_image

    out_dir = args.out_dir or (project_root / "Reports" / "image_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    image_paths = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts])

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    pipe = ImagePredictionPipeline(models_dir)
    pipe.load()

    rows: list[dict] = []

    for i, path in enumerate(image_paths, start=1):
        true_label = infer_label_from_path(path)
        print(f"[{i}/{len(image_paths)}] {path.name}")

        res = extract_ecg_from_image(str(path))

        row = {
            "path": str(path),
            "file": path.name,
            "true_label": "" if true_label is None else int(true_label),
            "extract_ok": bool(res.ok),
            "extract_msg": res.msg,
            "prob_abnormal": "",
            "pred_label": "",
            "threshold": float(args.threshold),
        }

        if res.ok:
            try:
                prob = pipe.predict_image_probability(res.beat_187)
                pred = 1 if prob >= float(args.threshold) else 0
                row["prob_abnormal"] = float(prob)
                row["pred_label"] = int(pred)
            except Exception as e:
                row["extract_ok"] = False
                row["extract_msg"] = f"Prediction failed: {e}"

        rows.append(row)

    metrics = compute_metrics(rows, threshold=float(args.threshold))

    csv_path = out_dir / "image_eval_predictions.csv"
    json_path = out_dir / "image_eval_metrics.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "file",
                "true_label",
                "extract_ok",
                "extract_msg",
                "prob_abnormal",
                "pred_label",
                "threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    print("\n========== IMAGE SOURCE EVALUATION ==========")
    print(f"Images total:          {metrics.n_images}")
    print(f"Labeled images:        {metrics.n_labeled}")
    print(f"Extracted OK:          {metrics.n_extracted_ok}")
    print(f"Extraction failed:     {metrics.n_extracted_fail}")
    print(f"Extraction success:    {metrics.extraction_success_rate * 100:.1f}%")
    print(f"Classified labeled:    {metrics.n_classified_labeled}")

    if metrics.accuracy is not None:
        print("")
        print(f"Accuracy:              {metrics.accuracy:.4f}")
        print(f"Precision:             {metrics.precision if metrics.precision is not None else 'n/a'}")
        print(f"Recall:                {metrics.recall if metrics.recall is not None else 'n/a'}")
        print(f"Specificity:           {metrics.specificity if metrics.specificity is not None else 'n/a'}")
        print(f"F1:                    {metrics.f1 if metrics.f1 is not None else 'n/a'}")
        print(f"Confusion:             TP={metrics.tp} TN={metrics.tn} FP={metrics.fp} FN={metrics.fn}")
    else:
        print("\nNo labeled extracted images available for classification metrics.")

    print("")
    print(f"Saved predictions CSV: {csv_path}")
    print(f"Saved metrics JSON:    {json_path}")


if __name__ == "__main__":
    main()