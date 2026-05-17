"""SARIMAX forecast model — giá phòng theo stay_date.

Mỗi (hotel_id, room_type) là một series độc lập:
    y[d] = price từ snapshot mới nhất của stay_date d
    exog (X): is_holiday, is_weekend, dow_sin/cos, month_sin/cos

Hyperparameter: pmdarima.auto_arima với:
    seasonal=True, m=7         (weekly seasonality)
    max_p=3, max_q=3, max_P=2, max_Q=2
    stepwise=True              (nhanh hơn full grid)

Output: median forecast + 80% CI (alpha=0.2) — native từ SARIMAX.

Sẽ điền ở Bước 3 (ROADMAP §3 Bước 3).
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import joblib
import pandas as pd


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


class ForecastModel:
    """SARIMAX wrapper. fit() trên một series; predict() ra DataFrame [p10, p50, p90].

    TODO Bước 3: hoàn thiện fit/predict + CI.
    """

    def __init__(self) -> None:
        self.model = None
        self.hotel_id: Optional[int] = None
        self.room_type: Optional[str] = None

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
        idx = index if index is not None else pd.RangeIndex(n_periods)
        return ForecastResult(
            index=idx,
            p10=pd.Series(conf_int[:, 0], index=idx),
            p50=pd.Series(forecast, index=idx),
            p90=pd.Series(conf_int[:, 1], index=idx),
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "ForecastModel":
        return joblib.load(Path(path))
