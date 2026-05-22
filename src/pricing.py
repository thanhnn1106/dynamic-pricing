"""Pricing optimization — ghép forecast + demand.

Cho mỗi stay_date và bối cảnh:
    1. Forecast cho ra [p10, p50, p90] → price grid trong [p10*0.7, p90*1.3]
    2. Với mỗi giá X trên grid, demand model dự đoán P(book | X, context)
    3. expected_revenue(X) = X × P(book | X)
    4. optimal_price = argmax expected_revenue
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .demand import DemandModel, CATEGORICAL_FEATURES
from .forecast import ForecastResult


@dataclass
class PricingCurve:
    """Kết quả pricing optimization cho 1 stay_date."""
    price_grid: np.ndarray
    prob_book: np.ndarray
    expected_revenue: np.ndarray
    optimal_price: float
    optimal_prob: float
    max_revenue: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "price": self.price_grid,
            "prob_book": self.prob_book,
            "expected_revenue": self.expected_revenue,
        })


def _prepare_X(rows_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Prepare X matrix khớp `demand_model.feature_cols`.

    Cùng logic encoding như `demand.prepare_X_y` nhưng cho inference path
    (không filter NA, không return y). `reindex` cuối đảm bảo:
    - Đúng thứ tự cột như khi train
    - Missing dummies (vd row chỉ có segment='Entry' → không có dummy nào)
      được fill 0
    """
    df = rows_df.copy()
    numeric = [c for c in feature_cols if not c.startswith("room_type_segment_")]
    X_num = df[numeric].astype(float)
    X_cat = pd.get_dummies(
        df[CATEGORICAL_FEATURES],
        prefix=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=float,
    )
    X = pd.concat(
        [X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1
    )
    return X.reindex(columns=feature_cols, fill_value=0.0)


def optimize_price(
    features_row: pd.Series | pd.DataFrame,
    forecast: ForecastResult,
    demand: DemandModel,
    *,
    grid_size: int = 80,
    grid_low_mult: float = 0.7,
    grid_high_mult: float = 1.3,
) -> PricingCurve:
    """Tìm giá tối ưu cho MỘT stay_date.

    Args:
        features_row: 1 snapshot context (Series hoặc 1-row DataFrame) với mọi
            feature columns trừ `price` (sẽ được vary).
        forecast: kết quả forecast cho 1 stay_date — `p10.iloc[0]` và
            `p90.iloc[0]` được dùng để bound `price_grid`.
        demand: trained `DemandModel` (LogReg hoặc LGBM).
        grid_size: số điểm grid (default 80).
        grid_low_mult, grid_high_mult: hệ số mở rộng từ CI band. Default
            [0.7×p10, 1.3×p90] (theo ROADMAP §3 Bước 5).

    Returns:
        PricingCurve với optimal_price, max_revenue, full curves.
    """
    low = float(forecast.p10.iloc[0]) * grid_low_mult
    high = float(forecast.p90.iloc[0]) * grid_high_mult
    if low <= 0 or high <= low:
        raise ValueError(
            f"Invalid price grid bounds: low={low}, high={high}. "
            "Kiểm tra forecast.p10/p90 và multipliers."
        )
    price_grid = np.linspace(low, high, grid_size)

    # Normalize input → 1-row DataFrame
    if isinstance(features_row, pd.Series):
        base = features_row.to_frame().T
    else:
        base = features_row.head(1).reset_index(drop=True)

    # Build candidates: lặp base, thay price
    candidates = pd.concat(
        [base.assign(price=p) for p in price_grid], ignore_index=True
    )
    X = _prepare_X(candidates, demand.feature_cols)
    probs = demand.predict_proba(X)
    exp_rev = price_grid * probs

    i = int(exp_rev.argmax())
    return PricingCurve(
        price_grid=price_grid,
        prob_book=probs,
        expected_revenue=exp_rev,
        optimal_price=float(price_grid[i]),
        optimal_prob=float(probs[i]),
        max_revenue=float(exp_rev[i]),
    )
