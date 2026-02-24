from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BeatSample:
    beat: np.ndarray
    true_label: int | None
    index: int
    source: str  # "PTBDB" / "CSV" / "IMAGE" / "SERIAL"


def ensure_187(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size > 187:
        x = x[:187]
    if x.size < 187:
        x = np.pad(x, (0, 187 - x.size))
    return x


class PTBDBSource:
    """
    Reads:
      Dataset/ptbdb_normal.csv
      Dataset/ptbdb_abnormal.csv

    Handles both:
      - 187 columns (beats only)
      - 188 columns (187 samples + label)
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.df: pd.DataFrame | None = None
        self.i = 0

    def load(self):
        n = self.dataset_dir / "ptbdb_normal.csv"
        a = self.dataset_dir / "ptbdb_abnormal.csv"
        if not n.exists() or not a.exists():
            raise FileNotFoundError("Missing Dataset/ptbdb_normal.csv or Dataset/ptbdb_abnormal.csv")

        df_n = pd.read_csv(n, header=None)
        df_a = pd.read_csv(a, header=None)

        def add_label(df: pd.DataFrame, fallback_label: int) -> pd.DataFrame:
            df = df.copy()
            y = None
            if df.shape[1] >= 188:
                last = pd.to_numeric(df.iloc[:, 187], errors="coerce")
                uniq = set(last.dropna().unique().tolist())
                if uniq.issubset({0, 1}):
                    y = last.astype(int)
            df["__label__"] = y if y is not None else int(fallback_label)
            return df

        df_n = add_label(df_n, 0)
        df_a = add_label(df_a, 1)

        self.df = (
            pd.concat([df_n, df_a], ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        self.i = 0

    def next(self) -> BeatSample:
        if self.df is None:
            self.load()
        assert self.df is not None

        if self.i >= len(self.df):
            self.i = 0

        row = self.df.iloc[self.i]
        idx = self.i
        self.i += 1

        beat = ensure_187(row.iloc[:187].astype(float).to_numpy())
        true_label = int(row["__label__"]) if "__label__" in row else None
        return BeatSample(beat=beat, true_label=true_label, index=idx, source="PTBDB")


class CSVSource:
    """
    CSV can be:
      - 187 columns (beats only)
      - 188 columns (187 samples + label)

    IMPORTANT:
    - X ALWAYS = first 187 columns only (prevents leakage -> fake 100% accuracy)
    - y ONLY read from column 187 if it is clearly {0,1}
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.i = 0

    def load(self):
        df = pd.read_csv(self.csv_path, header=None)

        # Features: first 187 columns only
        X = df.iloc[:, :187].to_numpy(dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Labels: column 187 only if valid binary labels
        self.y = None
        if df.shape[1] >= 188:
            ycol = pd.to_numeric(df.iloc[:, 187], errors="coerce")
            uniq = set(ycol.dropna().unique().tolist())
            if uniq.issubset({0, 1}):
                self.y = ycol.astype(int).to_numpy()

        self.X = X
        self.i = 0

    def next(self) -> BeatSample:
        if self.X is None:
            self.load()
        assert self.X is not None

        if self.i >= len(self.X):
            self.i = 0

        idx = self.i
        beat = ensure_187(self.X[self.i])

        true = None
        if self.y is not None and self.i < len(self.y):
            true = int(self.y[self.i])

        self.i += 1
        return BeatSample(beat=beat, true_label=true, index=idx, source="CSV")