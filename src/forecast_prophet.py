"""Prophet forecast model — alternative cho SARIMAX (Bước 3).

Mirror API của `src.forecast.ForecastModel` để Streamlit app dispatch
được giữa 2 backend dễ dàng.

Comparison findings (notebook 05_model_comparison.ipynb):
- Prophet MAPE ~9.5% vs SARIMAX ~27% — point forecast tốt hơn 2.8×
- Prophet CI quá hẹp với default — cần mcmc_samples để calibrated
- Train time với MCMC: ~30-60s/series (vs SARIMAX ~30s)

Serialization: dùng Prophet's `model_to_json` (pickle native không
đảm bảo cross-machine — Prophet docs khuyến nghị JSON).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .forecast import ForecastResult, _slugify

DEFAULT_FEATURES = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.parquet"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class ProphetForecastModel:
    """Prophet wrapper. fit() trên 1 series; predict() ra ForecastResult."""

    def __init__(self) -> None:
        self.model = None
        self.hotel_id: Optional[int] = None
        self.room_type: Optional[str] = None
        self.last_train_date: Optional[pd.Timestamp] = None

    def fit(
        self,
        series: pd.Series,
        exog: Optional[pd.DataFrame] = None,   # Prophet KHÔNG cần (built-in seasonality + holidays). Giữ param cho API parity.
        hotel_id: Optional[int] = None,
        room_type: Optional[str] = None,
        mcmc_samples: int = 0,
        interval_width: float = 0.8,
    ) -> "ProphetForecastModel":
        from prophet import Prophet

        self.hotel_id = hotel_id
        self.room_type = room_type
        self.last_train_date = series.index[-1]

        dfp = pd.DataFrame({"ds": series.index, "y": series.values})
        self.model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=interval_width,
            mcmc_samples=mcmc_samples,
        )
        self.model.add_country_holidays(country_name="VN")
        self.model.fit(dfp)
        return self

    def predict(
        self,
        n_periods: int,
        exog_future: Optional[pd.DataFrame] = None,   # ignored (parity)
        alpha: float = 0.2,    # ignored — Prophet's interval_width set tại fit
        index: Optional[pd.DatetimeIndex] = None,
    ) -> ForecastResult:
        if self.model is None:
            raise RuntimeError("Model chưa fit. Gọi .fit() trước.")
        if index is None:
            start = self.last_train_date + pd.Timedelta(days=1)
            index = pd.date_range(start, periods=n_periods, freq="D")

        future = pd.DataFrame({"ds": index})
        fc = self.model.predict(future)
        return ForecastResult(
            index=index,
            p10=pd.Series(fc["yhat_lower"].values, index=index),
            p50=pd.Series(fc["yhat"].values, index=index),
            p90=pd.Series(fc["yhat_upper"].values, index=index),
        )

    def save(self, path: str | Path) -> None:
        """Dùng Prophet's model_to_json (pickle native không reliable cross-machine)."""
        from prophet.serialize import model_to_json
        state = {
            "hotel_id": self.hotel_id,
            "room_type": self.room_type,
            "last_train_date": self.last_train_date,
            "prophet_json": model_to_json(self.model) if self.model is not None else None,
        }
        joblib.dump(state, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "ProphetForecastModel":
        from prophet.serialize import model_from_json
        state = joblib.load(Path(path))
        instance = cls()
        instance.hotel_id = state["hotel_id"]
        instance.room_type = state["room_type"]
        instance.last_train_date = state["last_train_date"]
        if state["prophet_json"]:
            instance.model = model_from_json(state["prophet_json"])
        return instance


def model_path_prophet(hotel_id: int, room_type_name: str,
                        models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"forecast_prophet_{hotel_id}_{_slugify(room_type_name)}.joblib"


def train_all_prophet(
    features_path: Path | str | None = None,
    holdout_days: int = 30,
    target_lead_time: int = 30,
    tolerance: int = 5,
    mcmc_samples: int = 0,
    interval_width: float = 0.8,
    models_dir: Path | str = MODELS_DIR,
) -> pd.DataFrame:
    """Fit 1 Prophet per (hotel, room_type), evaluate trên holdout.

    `mcmc_samples=0` → default (fast, ~0.1s/series, CI hẹp).
    `mcmc_samples=300` → full Bayesian (slow ~30-60s/series, CI calibrated).
    """
    from .forecast import build_series

    df = pd.read_parquet(features_path or DEFAULT_FEATURES)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for hotel_id in sorted(df["hotel_id"].unique()):
        sub = df[df["hotel_id"] == hotel_id]
        for room_type in sorted(sub["room_type_name"].unique()):
            target, _ = build_series(
                df, int(hotel_id), room_type,
                target_lead_time=target_lead_time, tolerance=tolerance,
            )
            target = target.dropna()
            train_y = target.iloc[:-holdout_days]
            hold_y = target.iloc[-holdout_days:]

            m = ProphetForecastModel().fit(
                train_y,
                hotel_id=int(hotel_id), room_type=room_type,
                mcmc_samples=mcmc_samples, interval_width=interval_width,
            )
            fc = m.predict(holdout_days, index=hold_y.index)

            mape = float((np.abs(hold_y - fc.p50) / hold_y).mean())
            rmse = float(np.sqrt(((hold_y - fc.p50) ** 2).mean()))
            coverage = float(((hold_y >= fc.p10) & (hold_y <= fc.p90)).mean())

            path = model_path_prophet(int(hotel_id), room_type, models_dir)
            m.save(path)

            rows.append({
                "hotel_id": int(hotel_id),
                "room_type": room_type,
                "n_train": len(train_y),
                "n_hold": len(hold_y),
                "mape_pct": mape * 100,
                "rmse": rmse,
                "coverage80_pct": coverage * 100,
                "path": str(path.relative_to(Path(__file__).resolve().parents[1])),
            })
    return pd.DataFrame(rows)
