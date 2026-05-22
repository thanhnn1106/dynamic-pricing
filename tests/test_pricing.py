"""Unit tests cho `src.pricing.optimize_price`.

Test strategy: dùng real trained models (B3 + B4 artifacts) — đơn giản hơn
mock vì chỉ cần verify shape + invariant của output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.demand import DemandModel
from src.forecast import ForecastModel, ForecastResult, build_series, model_path
from src.pricing import PricingCurve, optimize_price

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

# Skip toàn module nếu chưa train models (CI fresh checkout).
pytestmark = pytest.mark.skipif(
    not FEATURES_PATH.exists()
    or not (MODELS_DIR / "demand_logreg.joblib").exists()
    or not (MODELS_DIR / "forecast_956_superior_city_view.joblib").exists(),
    reason="Cần features.parquet + trained models (chạy notebooks 02/03/04 trước).",
)


@pytest.fixture(scope="module")
def features_df():
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="module")
def sample_row(features_df):
    """1 snapshot từ test set — Superior City View, weekend, mid lead_time."""
    sub = features_df[
        (features_df["room_type_name"] == "Superior City View")
        & (features_df["is_weekend"] == 1)
        & (features_df["lead_time_days"].between(20, 40))
    ]
    return sub.iloc[0]


@pytest.fixture(scope="module")
def forecast_result(features_df):
    """Forecast 1 ngày cho Superior City View — dùng làm input cho optimize."""
    model = ForecastModel.load(model_path(956, "Superior City View"))
    target, exog = build_series(features_df, 956, "Superior City View",
                                 target_lead_time=30, tolerance=5)
    target = target.dropna()
    last_exog = exog.loc[target.index[-1:]]   # 1 row exog
    return model.predict(n_periods=1, exog_future=last_exog,
                         index=pd.DatetimeIndex([target.index[-1]]))


@pytest.fixture(scope="module")
def demand_logreg():
    return DemandModel.load(MODELS_DIR / "demand_logreg.joblib")


@pytest.fixture(scope="module")
def demand_lgbm():
    return DemandModel.load(MODELS_DIR / "demand_lgbm.joblib")


# ---------- Tests ----------------------------------------------------------


def test_returns_pricing_curve(sample_row, forecast_result, demand_logreg):
    out = optimize_price(sample_row, forecast_result, demand_logreg)
    assert isinstance(out, PricingCurve)


def test_grid_size_default_80(sample_row, forecast_result, demand_logreg):
    out = optimize_price(sample_row, forecast_result, demand_logreg)
    assert len(out.price_grid) == 80
    assert len(out.prob_book) == 80
    assert len(out.expected_revenue) == 80


def test_grid_size_custom(sample_row, forecast_result, demand_logreg):
    out = optimize_price(sample_row, forecast_result, demand_logreg, grid_size=20)
    assert len(out.price_grid) == 20


def test_optimal_price_in_grid(sample_row, forecast_result, demand_logreg):
    """Quan trọng: optimal_price phải là 1 trong các giá trị của price_grid,
    không phải interpolate ngoài."""
    out = optimize_price(sample_row, forecast_result, demand_logreg)
    assert out.optimal_price in out.price_grid


def test_max_revenue_is_argmax(sample_row, forecast_result, demand_logreg):
    """max_revenue phải = max(expected_revenue)."""
    out = optimize_price(sample_row, forecast_result, demand_logreg)
    assert out.max_revenue == out.expected_revenue.max()
    assert out.max_revenue == out.optimal_price * out.optimal_prob


def test_grid_uses_forecast_bounds(sample_row, forecast_result, demand_logreg):
    """Grid bound theo forecast.p10 × low_mult, p90 × high_mult."""
    out = optimize_price(sample_row, forecast_result, demand_logreg,
                          grid_low_mult=0.5, grid_high_mult=2.0)
    expected_low = float(forecast_result.p10.iloc[0]) * 0.5
    expected_high = float(forecast_result.p90.iloc[0]) * 2.0
    assert out.price_grid[0] == pytest.approx(expected_low)
    assert out.price_grid[-1] == pytest.approx(expected_high)


def test_probs_in_unit_interval(sample_row, forecast_result, demand_logreg):
    out = optimize_price(sample_row, forecast_result, demand_logreg)
    assert (out.prob_book >= 0).all()
    assert (out.prob_book <= 1).all()


def test_works_with_lgbm(sample_row, forecast_result, demand_lgbm):
    """API agnostic về kind của demand model."""
    out = optimize_price(sample_row, forecast_result, demand_lgbm)
    assert out.optimal_price > 0
    assert 0 <= out.optimal_prob <= 1


def test_accepts_dataframe_input(sample_row, forecast_result, demand_logreg):
    """features_row có thể là Series hoặc 1-row DataFrame."""
    out_series = optimize_price(sample_row, forecast_result, demand_logreg)
    out_df = optimize_price(sample_row.to_frame().T, forecast_result, demand_logreg)
    np.testing.assert_allclose(out_series.price_grid, out_df.price_grid)
    np.testing.assert_allclose(out_series.prob_book, out_df.prob_book)


def test_logreg_demand_curve_monotonic_decreasing(sample_row, forecast_result, demand_logreg):
    """LogReg là model tuyến tính → P(book) phải monotonic decreasing theo
    price (vì coef của price là âm)."""
    out = optimize_price(sample_row, forecast_result, demand_logreg)
    diffs = np.diff(out.prob_book)
    assert (diffs <= 1e-9).all(), "LogReg P(book) phải không tăng theo price"


def test_to_frame_shape(sample_row, forecast_result, demand_logreg):
    out = optimize_price(sample_row, forecast_result, demand_logreg)
    df = out.to_frame()
    assert df.shape == (80, 3)
    assert list(df.columns) == ["price", "prob_book", "expected_revenue"]


def test_negative_p10_uses_p50_anchor(sample_row, demand_logreg):
    """SARIMAX extrapolate → p10 âm. Optimizer phải dùng p50 làm anchor,
    floor low ≥ p50 × 0.3."""
    idx = pd.DatetimeIndex(["2026-08-01"])
    fake_forecast = ForecastResult(
        index=idx,
        p10=pd.Series([-100.0], index=idx),
        p50=pd.Series([1000.0], index=idx),
        p90=pd.Series([2000.0], index=idx),
    )
    out = optimize_price(sample_row, fake_forecast, demand_logreg)
    assert out.price_grid[0] == 1000.0 * 0.3    # p50 anchor kicked in
    assert out.price_grid[-1] == 2000.0 * 1.3   # p90 × 1.3 still wins (positive)


def test_both_p10_p90_negative_uses_p50_anchor(sample_row, demand_logreg):
    """Cả p10 và p90 âm nhưng p50 dương → vẫn build được grid quanh p50."""
    idx = pd.DatetimeIndex(["2026-08-01"])
    fake_forecast = ForecastResult(
        index=idx,
        p10=pd.Series([-2000.0], index=idx),
        p50=pd.Series([500.0], index=idx),
        p90=pd.Series([-100.0], index=idx),
    )
    out = optimize_price(sample_row, fake_forecast, demand_logreg)
    assert out.price_grid[0] == 500.0 * 0.3     # 150
    assert out.price_grid[-1] == 500.0 * 1.7    # 850


def test_p50_nonpositive_raises(sample_row, demand_logreg):
    """p50 <= 0 → forecast vô nghĩa, raise (không có anchor để fallback)."""
    idx = pd.DatetimeIndex(["2026-08-01"])
    fake_forecast = ForecastResult(
        index=idx,
        p10=pd.Series([-200.0], index=idx),
        p50=pd.Series([-100.0], index=idx),
        p90=pd.Series([-50.0], index=idx),
    )
    with pytest.raises(ValueError, match="p50"):
        optimize_price(sample_row, fake_forecast, demand_logreg)
