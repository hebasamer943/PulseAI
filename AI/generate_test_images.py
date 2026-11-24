import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "Dataset/mitbih_test.csv"
OUT_DIR = "TestImages"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, header=None)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(int)

# ✅ خليها Balanced:
N_NORMAL = 100
N_ABNORMAL = 100

normal_idx = np.where(y == 0)[0]
abnormal_idx = np.where(y != 0)[0]

# اختيارات عشوائية
np.random.seed(42)
pick_normal = np.random.choice(normal_idx, size=min(N_NORMAL, len(normal_idx)), replace=False)
pick_abn = np.random.choice(abnormal_idx, size=min(N_ABNORMAL, len(abnormal_idx)), replace=False)

picked = np.concatenate([pick_normal, pick_abn])
np.random.shuffle(picked)

for k, i in enumerate(picked):
    sig = X[i]
    label = int(y[i])

    sig = sig - np.median(sig)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)

    plt.figure(figsize=(8, 2.2))
    plt.plot(sig, linewidth=3)
    plt.grid(True, alpha=0.35)
    plt.ylim(-1.2, 1.2)
    plt.axis("off")

    # نسميها Normal / Abnormal حتى تميزّي بسهولة
    cls = "normal" if label == 0 else "abnormal"
    name = f"{cls}_{k:04d}_origlabel_{label}.png"
    plt.savefig(os.path.join(OUT_DIR, name), dpi=160, bbox_inches="tight", pad_inches=0.02)
    plt.close()

print(f"✅ Done. Saved {len(picked)} images in: {OUT_DIR}")