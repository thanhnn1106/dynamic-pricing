"""Feature engineering.

Sinh các feature từ raw inventory (đã `clean()`):
- Lead time: `lead_time_days` + bucket
- Calendar: dow, month, is_weekend, is_holiday (VN), sin/cos encoding
- Inventory state: occupancy_pct, available_pct
- Demand label: `did_book` (1 nếu total_booked tăng ở snapshot kế)

Đầu ra: `data/processed/features.parquet` — model-ready table cho cả forecast
(Bước 3) và demand model (Bước 4).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.parquet"

# Lazy-loaded VN holiday set (init lần đầu, dùng lại cho mọi call)
_VN_HOLIDAYS: set | None = None


def _vn_holidays(years: range) -> set:
    """Trả về set ngày lễ VN. Cache module-level."""
    global _VN_HOLIDAYS
    if _VN_HOLIDAYS is None:
        import holidays
        _VN_HOLIDAYS = set(holidays.country_holidays("VN", years=list(years)).keys())
    return _VN_HOLIDAYS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm feature + label columns. Input: df đã qua `data.clean()`."""
    df = df.copy()

    # --- Lead time -----------------------------------------------------------
    df["lead_time_days"] = (df["date"] - df["updated_date"]).dt.days
    # Bins phù hợp domain hotel: same-day / <1 tuần / <1 tháng / <3 tháng / xa.
    # Sau clean() thì lead_time >= 0; bin [-1, 1) bắt giá trị 0 (same-day).
    df["lead_time_bucket"] = pd.cut(
        df["lead_time_days"],
        bins=[-1, 1, 7, 30, 90, 365],
        labels=["same", "<1w", "<1m", "<3m", ">3m"],
    )

    # --- Calendar ------------------------------------------------------------
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 2)
    vn = _vn_holidays(years)
    df["is_holiday"] = df["date"].dt.date.isin(vn).astype(int)

    # Cyclic encoding giúp model tuyến tính (LogReg, SARIMAX exog) nhận biết
    # tính chu kỳ: Mon-Sun và Jan-Dec đều cyclic, không phải ordinal.
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Inventory ratios ----------------------------------------------------
    # np.where tránh ZeroDivision nếu `total == 0` (room đóng hẳn).
    df["occupancy_pct"] = np.where(df["total"] > 0, df["total_booked"] / df["total"], np.nan)
    df["available_pct"] = np.where(df["total"] > 0, df["available"] / df["total"], np.nan)

    # --- Demand label `did_book` ---------------------------------------------
    # Pair 2 snapshot liên tiếp của cùng (hotel, room_type, stay_date), ordered
    # theo updated_date. Nếu total_booked tăng → có booking xảy ra giữa 2
    # snapshot → did_book = 1. Snapshot cuối mỗi group không có "next" → NA
    # (sẽ bị drop khi train demand model).
    df = df.sort_values(
        ["hotel_id", "room_type_name", "date", "updated_date"]
    ).reset_index(drop=True)
    grp = df.groupby(["hotel_id", "room_type_name", "date"], sort=False)
    df["next_total_booked"] = grp["total_booked"].shift(-1)
    df["did_book"] = (df["next_total_booked"] > df["total_booked"]).astype("Int64")
    df.loc[df["next_total_booked"].isna(), "did_book"] = pd.NA

    return df


def save_features(df: pd.DataFrame, path: Path = DEFAULT_OUT) -> Path:
    """Persist features → parquet. Tự tạo parent dir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def prepare_features(csv_path: Path | str | None = None,
                     out_path: Path | str = DEFAULT_OUT) -> pd.DataFrame:
    """One-shot pipeline: load_raw → clean → build_features → save_features."""
    from .data import load_raw, clean

    df = load_raw() if csv_path is None else load_raw(csv_path)
    df = clean(df)
    df = build_features(df)
    save_features(df, out_path)
    return df
