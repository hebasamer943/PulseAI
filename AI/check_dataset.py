from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parents[1]
data_dir = base / "Dataset"

files = [
    "mitbih_train.csv",
    "mitbih_test.csv",
    "ptbdb_normal.csv",
    "ptbdb_abnormal.csv",
]

for f in files:
    p = data_dir / f
    if not p.exists():
        print(f"[MISSING] {f}")
        continue

    df = pd.read_csv(p, header=None)
    print(f"\n[OK] {f}")
    print("shape =", df.shape)
    print("first row (last 5 values) =", df.iloc[0, -5:].to_list())
    print("last column unique sample =", df.iloc[:20, -1].unique())

