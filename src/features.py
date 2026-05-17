"""Feature engineering.

Sinh các feature từ raw inventory:
- Lead time: (date - updated_date).dt.days, kèm bucket
- Calendar: dow, month, is_weekend, is_holiday (VN), sin/cos encoding
- Inventory state: occupancy_pct, available_pct
- Lag/rolling (cho demand model — KHÔNG dùng làm exog của SARIMA forecast)

Sẽ điền ở Bước 2 (ROADMAP §3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lazy-loaded VN holiday set
_VN_HOLIDAYS: set | None = None


def _vn_holidays(years: range) -> set:
    """Trả về set ngày lễ VN. Cache module-level."""
    global _VN_HOLIDAYS
    if _VN_HOLIDAYS is None:
        import holidays
        _VN_HOLIDAYS = set(holidays.country_holidays("VN", years=list(years)).keys())
    return _VN_HOLIDAYS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sinh feature columns. TODO: hoàn thiện ở Bước 2."""
    df = df.copy()
    # Lead time
    df["lead_time_days"] = (df["date"] - df["updated_date"]).dt.days

    # Calendar
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 2)
    vn = _vn_holidays(years)
    df["is_holiday"] = df["date"].dt.date.isin(vn).astype(int)

    # Cyclic encoding cho dow/month (giúp SARIMAX và LGBM nhận biết tính chu kỳ)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Inventory
    df["occupancy_pct"] = df["total_booked"] / df["total"]
    df["available_pct"] = df["available"] / df["total"]

    # TODO Bước 2: lag price 7/14 ngày, rolling mean 14/28, target encoding room_type
    return df
