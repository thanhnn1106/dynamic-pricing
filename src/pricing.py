"""Pricing optimization — ghép forecast + demand.

Cho mỗi stay_date và bối cảnh:
    1. Forecast cho ra [p10, p50, p90] => price grid trong khoảng [p10*0.7, p90*1.3]
    2. Với mỗi giá X trên grid, demand model dự đoán P(book | X)
    3. expected_revenue(X) = X × P(book | X)
    4. optimal_price = argmax expected_revenue

Sẽ điền ở Bước 5 (ROADMAP §3 Bước 5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .demand import DemandModel
from .forecast import ForecastResult


@dataclass
class PricingCurve:
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


def optimize_price(
    features_row: pd.Series,
    forecast: ForecastResult,
    demand: DemandModel,
    *,
    grid_size: int = 80,
    grid_low_mult: float = 0.7,
    grid_high_mult: float = 1.3,
) -> PricingCurve:
    """Tìm giá tối ưu cho MỘT stay_date. TODO Bước 5."""
    low = float(forecast.p10.iloc[0]) * grid_low_mult
    high = float(forecast.p90.iloc[0]) * grid_high_mult
    price_grid = np.linspace(low, high, grid_size)

    rows = []
    base = features_row.copy()
    for p in price_grid:
        x = base.copy()
        x["price"] = p
        rows.append(x)
    X = pd.DataFrame(rows)[demand.feature_cols]  # đảm bảo cùng cột

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
