"""Demand model — P(book | price, context).

Target binary `did_book` đã được tạo ở Bước 2 (`src/features.py`):
    1 nếu `total_booked` tăng giữa 2 snapshot liên tiếp của cùng
    `(hotel_id, room_type_name, stay_date)`, sort theo `updated_date`.
    NA cho snapshot cuối mỗi group (không có "next").

Baseline: `sklearn.linear_model.LogisticRegression` (interpretable — signed
coefficient của `price` cho ta direction của elasticity).
Production: `lightgbm.LGBMClassifier` (capture non-linear, interaction).

Caveat (ROADMAP §3 Bước 4): data chỉ có giá thực tế đã set — không có
counterfactual. Model học correlation, chưa hẳn causal. Predictive, not
causal price elasticity. Phase 2 cần A/B test để validate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

DEFAULT_FEATURES = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.parquet"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Features cho demand model. KHÔNG include raw `dow`, `month` (đã có sin/cos).
# KHÔNG include raw `available` (correlated với `total`, đã có `available_pct`).
NUMERIC_FEATURES = [
    "price",                              # KEY — biến quyết định
    "occupancy_pct", "available_pct",     # inventory state
    "lead_time_days",                     # khoảng cách đến stay_date
    "is_weekend", "is_holiday",           # calendar binary
    "dow_sin", "dow_cos",                 # weekly cyclic
    "month_sin", "month_cos",             # monthly cyclic
]
CATEGORICAL_FEATURES = ["room_type_segment"]   # Entry / Mid / High → 1-hot


def prepare_X_y(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build `(X, y)` matrix từ features panel.

    - Drop snapshot cuối mỗi group (did_book NA).
    - One-hot encode `room_type_segment` (`drop_first=True` tránh
      multicollinearity với LogReg).
    """
    df = features_df.dropna(subset=["did_book"]).copy()

    cols = list(feature_cols) if feature_cols is not None else list(NUMERIC_FEATURES)
    X_num = df[cols].astype(float)
    X_cat = pd.get_dummies(
        df[CATEGORICAL_FEATURES],
        prefix=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    y = df["did_book"].astype(int).reset_index(drop=True)
    return X, y


def time_split(
    features_df: pd.DataFrame,
    holdout_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split theo `updated_date` (THỜI ĐIỂM PREDICT, không phải
    `date` = stay_date).

    Demand model dùng cho Sale team: tại time T họ predict booking xảy ra
    trong window snapshot kế tiếp. Train = snapshot chụp ≤ cutoff, test =
    snapshot chụp > cutoff → mô phỏng deployment thực.

    Note: split theo `date` (stay_date) bị problem — test = stay_dates xa
    tương lai, snapshots toàn lead_time >60d, gần như không có booking
    event → AUC undefined.
    """
    cutoff = features_df["updated_date"].max() - pd.Timedelta(days=holdout_days)
    train = features_df[features_df["updated_date"] <= cutoff]
    test = features_df[features_df["updated_date"] > cutoff]
    return train, test


class DemandModel:
    """Classifier wrapper: `fit(X, y)` → `predict_proba(X) -> P(book)`."""

    def __init__(self, kind: str = "logreg") -> None:
        if kind not in ("logreg", "lgbm"):
            raise ValueError(f"Unknown kind={kind!r}; use 'logreg' or 'lgbm'.")
        self.kind = kind
        self.model = None
        self.feature_cols: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DemandModel":
        self.feature_cols = list(X.columns)
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            # class_weight='balanced' xử lý imbalance ~6.8% positive.
            # StandardScaler cần thiết: price ~10^6, ratios ~10^0.
            self.model = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                )),
            ])
        else:  # lgbm
            from lightgbm import LGBMClassifier
            self.model = LGBMClassifier(
                objective="binary",
                is_unbalance=True,    # tương đương class_weight='balanced'
                learning_rate=0.05,
                n_estimators=400,
                num_leaves=31,        # giảm từ 63 — small dataset, tránh overfit
                min_child_samples=30,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
            )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Trả về vector `P(did_book=1)`."""
        if self.model is None:
            raise RuntimeError("Model chưa fit.")
        return self.model.predict_proba(X)[:, 1]

    def coefficients(self) -> pd.Series | None:
        """LogReg only — return signed coefficient per feature (post-scale).

        Note: vì StandardScaler bên trong pipeline, coefficient này là
        \"per 1 std deviation of feature\", không phải raw unit. Vẫn dùng
        được để đọc DIRECTION (sign) của elasticity.
        """
        if self.kind != "logreg" or self.model is None:
            return None
        coef = self.model.named_steps["clf"].coef_[0]
        return pd.Series(coef, index=self.feature_cols).sort_values()

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "DemandModel":
        return joblib.load(Path(path))


def train_all(
    features_path: Path | str | None = None,
    holdout_days: int = 30,
    models_dir: Path | str = MODELS_DIR,
) -> pd.DataFrame:
    """Fit LogReg + LightGBM, evaluate trên holdout time-based.

    Trả về metrics DataFrame (AUC, average_precision, Brier).
    Saves: `models/demand_logreg.joblib`, `models/demand_lgbm.joblib`.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
    )

    df = pd.read_parquet(features_path or DEFAULT_FEATURES)
    train_df, test_df = time_split(df, holdout_days=holdout_days)

    X_train, y_train = prepare_X_y(train_df)
    X_test, y_test = prepare_X_y(test_df)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for kind in ["logreg", "lgbm"]:
        model = DemandModel(kind=kind).fit(X_train, y_train)
        proba_train = model.predict_proba(X_train)
        proba_test = model.predict_proba(X_test)
        path = models_dir / f"demand_{kind}.joblib"
        model.save(path)
        rows.append({
            "kind": kind,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "pos_rate_train": float(y_train.mean()),
            "pos_rate_test": float(y_test.mean()),
            "auc_train": roc_auc_score(y_train, proba_train),
            "auc_test": roc_auc_score(y_test, proba_test),
            "avg_prec_test": average_precision_score(y_test, proba_test),
            "brier_test": brier_score_loss(y_test, proba_test),
            "path": str(path.relative_to(Path(__file__).resolve().parents[1])),
        })
    return pd.DataFrame(rows)
