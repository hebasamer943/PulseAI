from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Dataset"
OUT = DATASET / "mitbih_demo_mixed.csv"

df = pd.read_csv(DATASET / "mitbih_test.csv", header=None)

# labels: 0=N, 1=S, 2=V, 3=F, 4=Q
parts = []

for label, n in [
    (0, 30),  # Normal
    (1, 10),  # S
    (2, 10),  # V
    (3, 10),  # F
]:
    part = df[df.iloc[:, 187] == label].head(n)
    parts.append(part)

demo = pd.concat(parts, ignore_index=True)
demo = demo.sample(frac=1, random_state=42).reset_index(drop=True)

demo.to_csv(OUT, header=False, index=False)

print("Saved:", OUT)
print(demo.iloc[:, 187].value_counts().sort_index())