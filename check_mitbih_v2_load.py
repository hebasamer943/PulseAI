from pathlib import Path
import json
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "Models"

files = [
    "mitbih_stage1_binary_v2.keras",
    "mitbih_stage1_binary_v2_meta.json",
    "mitbih_stage2_subtypes_svf_v2.keras",
    "mitbih_stage2_subtypes_svf_v2_meta.json",
]

print("Checking files...")
for name in files:
    path = MODELS / name
    print(name, "FOUND" if path.exists() else "MISSING")

print("\nLoading Stage 1 v2...")
stage1 = tf.keras.models.load_model(MODELS / "mitbih_stage1_binary_v2.keras", compile=False)
print("Stage 1 loaded OK")

print("\nLoading Stage 2 v2...")
stage2 = tf.keras.models.load_model(MODELS / "mitbih_stage2_subtypes_svf_v2.keras", compile=False)
print("Stage 2 loaded OK")

print("\nReading meta files...")
m1 = json.loads((MODELS / "mitbih_stage1_binary_v2_meta.json").read_text(encoding="utf-8"))
m2 = json.loads((MODELS / "mitbih_stage2_subtypes_svf_v2_meta.json").read_text(encoding="utf-8"))

print("Stage 1 threshold:", m1.get("best_threshold_from_val"))
print("Stage 2 confidence threshold:", m2.get("recommended_confidence_threshold"))

print("\nDONE: MIT-BIH v2 files are valid.")