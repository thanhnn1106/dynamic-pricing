"""Demand model — P(book | price, context).

Target binary `did_book` được build từ pair snapshot liên tiếp:

    Cho cùng (hotel_id, room_type_name, stay_date) sort theo updated_date,
    did_book[t] = 1 nếu total_booked[t+1] > total_booked[t], else 0.

Features: price, occupancy_pct, lead_time_days, dow_sin/cos, is_weekend,
          is_holiday, room_type_segment, available (numeric)

Baseline: sklearn.linear_model.LogisticRegression (interpretable)
Production: lightgbm.LGBMClassifier

Sẽ điền ở Bước 4 (ROADMAP §3 Bước 4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


def build_did_book_target(df: pd.DataFrame) -> pd.DataFrame:
    """Sinh cột did_book từ snapshot liên tiếp. TODO Bước 4."""
    df = df.sort_values(["hotel_id", "room_type_name", "date", "updated_date"]).copy()
    grp = df.groupby(["hotel_id", "room_type_name", "date"])
    df["next_total_booked"] = grp["total_booked"].shift(-1)
    df["delta_booked"] = df["next_total_booked"] - df["total_booked"]
    df["did_book"] = (df["delta_booked"] > 0).astype("Int8")
    # Drop dòng cuối mỗi group (không có next snapshot → target NaN)
    df = df.dropna(subset=["next_total_booked"])
    return df


class DemandModel:
    """Wrapper cho classifier. TODO Bước 4."""

    def __init__(self, kind: str = "logreg") -> None:
        self.kind = kind
        self.model = None
        self.feature_cols: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DemandModel":
        self.feature_cols = list(X.columns)
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            self.model = Pipeline([
                ("scale", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=1000)),
            ])
        elif self.kind == "lgbm":
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(
                objective="binary", learning_rate=0.05, n_estimators=400,
                num_leaves=63, min_child_samples=30,
            )
        else:
            raise ValueError(f"Unknown kind={self.kind!r}; use 'logreg' or 'lgbm'.")
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Trả về P(did_book=1)."""
        if self.model is None:
            raise RuntimeError("Model chưa fit.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "DemandModel":
        return joblib.load(Path(path))
