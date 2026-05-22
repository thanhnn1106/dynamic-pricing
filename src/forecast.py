"""SARIMAX forecast model — giá phòng theo stay_date.

Mỗi (hotel_id, room_type) là một series độc lập:
    y[d] = price từ snapshot mới nhất của stay_date d
    exog (X): is_holiday, is_weekend, dow_sin/cos, month_sin/cos

Hyperparameter: pmdarima.auto_arima với:
    seasonal=True, m=7         (weekly seasonality)
    max_p=3, max_q=3, max_P=2, max_Q=2
    stepwise=True              (nhanh hơn full grid)

Output: median forecast + 80% CI (alpha=0.2) — native từ SARIMAX.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Exog cho SARIMAX. KHÔNG dùng occupancy/available vì future values unknown
# tại predict time → calendar-only là constraint thực dụng.
EXOG_COLS = [
    "is_holiday",
    "is_weekend",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]

DEFAULT_FEATURES = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.parquet"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _slugify(s: str) -> str:
    """Filename-safe slug: lowercase, alnum + '_'."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@dataclass
class ForecastResult:
    """Kết quả predict cho n_periods ngày tới."""
    index: pd.DatetimeIndex   # stay_date
    p10: pd.Series
    p50: pd.Series           # median forecast
    p90: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"p10": self.p10.values, "p50": self.p50.values, "p90": self.p90.values},
            index=self.index,
        )


def build_series(
    features_df: pd.DataFrame,
    hotel_id: int,
    room_type_name: str,
    target_lead_time: int = 30,
    tolerance: int = 5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Trích `(target, exog)` cho 1 `(hotel, room_type)` từ panel features.

    Target = `price` của snapshot có `lead_time_days` gần `target_lead_time` nhất
    (trong tolerance ±`tolerance` ngày). Đây là "fixed lead-time" convention —
    apples-to-apples giữa train (stay đầu data) và holdout (stay cuối data),
    tránh right-censoring: late stay_dates không có "near-stay" snapshot vì
    `updated_date` dữ liệu kết thúc sớm.

    Mặc định lead_time=30 ngày — Sale team thường set giá ~1 tháng trước stay.

    Exog = `EXOG_COLS` lấy trực tiếp từ panel (calendar features chỉ phụ thuộc
    stay_date nên giá trị giống nhau ở mọi snapshot cùng date).
    """
    sub = features_df[
        (features_df["hotel_id"] == hotel_id)
        & (features_df["room_type_name"] == room_type_name)
    ].copy()
    sub["_lt_dist"] = (sub["lead_time_days"] - target_lead_time).abs()
    sub = sub[sub["_lt_dist"] <= tolerance]

    # Mỗi stay_date 1 snapshot — chọn cái gần target_lead_time nhất.
    picked = (
        sub.sort_values("_lt_dist")
           .groupby("date")
           .head(1)
           .sort_values("date")
           .set_index("date")
    )
    # asfreq('D') gắn freq vào DatetimeIndex (yêu cầu của SARIMAX). Nếu có
    # stay_date không có snapshot trong window tolerance → NaN ở vị trí đó.
    target = picked["price"].asfreq("D")
    exog = picked[EXOG_COLS].asfreq("D")
    return target, exog


class ForecastModel:
    """SARIMAX wrapper. fit() trên một series; predict() ra `ForecastResult`."""

    def __init__(self) -> None:
        self.model = None
        self.hotel_id: Optional[int] = None
        self.room_type: Optional[str] = None
        self.last_train_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        hotel_id: Optional[int] = None,
        room_type: Optional[str] = None,
    ) -> "ForecastModel":
        from pmdarima import auto_arima

        self.hotel_id = hotel_id
        self.room_type = room_type
        self.last_train_date = series.index[-1]
        self.model = auto_arima(
            series,
            X=exog,
            seasonal=True, m=7,
            d=None, D=None,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3, max_P=2, max_Q=2,
        )
        return self

    def predict(
        self,
        n_periods: int,
        exog_future: Optional[pd.DataFrame] = None,
        alpha: float = 0.2,        # 80% CI
        index: Optional[pd.DatetimeIndex] = None,
    ) -> ForecastResult:
        if self.model is None:
            raise RuntimeError("Model chưa fit. Gọi .fit() trước.")
        forecast, conf_int = self.model.predict(
            n_periods=n_periods,
            X=exog_future,
            return_conf_int=True,
            alpha=alpha,
        )
        if index is None:
            start = self.last_train_date + pd.Timedelta(days=1)
            index = pd.date_range(start, periods=n_periods, freq="D")
        return ForecastResult(
            index=index,
            p10=pd.Series(conf_int[:, 0], index=index),
            p50=pd.Series(np.asarray(forecast), index=index),
            p90=pd.Series(conf_int[:, 1], index=index),
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "ForecastModel":
        return joblib.load(Path(path))


def model_path(hotel_id: int, room_type_name: str,
               models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"forecast_{hotel_id}_{_slugify(room_type_name)}.joblib"


def train_all(
    features_path: Path | str | None = None,
    holdout_days: int = 30,
    target_lead_time: int = 30,
    tolerance: int = 5,
    models_dir: Path | str = MODELS_DIR,
) -> pd.DataFrame:
    """Fit 1 model per `(hotel, room_type)`, evaluate trên holdout cuối series.

    `target_lead_time` + `tolerance` forwarded vào `build_series()`.
    Trả về DataFrame metrics: arima order, MAPE, RMSE, coverage@80%, joblib path.
    """
    df = pd.read_parquet(features_path or DEFAULT_FEATURES)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for hotel_id in sorted(df["hotel_id"].unique()):
        sub = df[df["hotel_id"] == hotel_id]
        for room_type in sorted(sub["room_type_name"].unique()):
            target, exog = build_series(
                df, int(hotel_id), room_type,
                target_lead_time=target_lead_time, tolerance=tolerance,
            )
            # Drop NaN nếu stay_date không có snapshot trong tolerance window.
            valid = target.dropna().index
            target = target.loc[valid]
            exog = exog.loc[valid]

            train_y = target.iloc[:-holdout_days]
            train_x = exog.iloc[:-holdout_days]
            hold_y = target.iloc[-holdout_days:]
            hold_x = exog.iloc[-holdout_days:]

            model = ForecastModel().fit(
                train_y, exog=train_x,
                hotel_id=int(hotel_id), room_type=room_type,
            )
            fc = model.predict(holdout_days, exog_future=hold_x, index=hold_y.index)

            mape = float((np.abs(hold_y - fc.p50) / hold_y).mean())
            rmse = float(np.sqrt(((hold_y - fc.p50) ** 2).mean()))
            coverage = float(((hold_y >= fc.p10) & (hold_y <= fc.p90)).mean())

            path = model_path(int(hotel_id), room_type, models_dir)
            model.save(path)

            rows.append({
                "hotel_id": int(hotel_id),
                "room_type": room_type,
                "target_lead_time": target_lead_time,
                "n_train": len(train_y),
                "n_hold": len(hold_y),
                "arima_order": str(model.model.order),
                "seasonal_order": str(model.model.seasonal_order),
                "mape_pct": mape * 100,
                "rmse": rmse,
                "coverage80_pct": coverage * 100,
                "path": str(path.relative_to(Path(__file__).resolve().parents[1])),
            })
    return pd.DataFrame(rows)
