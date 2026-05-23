"""Tests cho `src.forecast` — build_series + ForecastModel predict shape."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecast import (
    EXOG_COLS, ForecastModel, ForecastResult, build_series, model_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

pytestmark = pytest.mark.skipif(
    not FEATURES_PATH.exists()
    or not (MODELS_DIR / "forecast_956_superior_city_view.joblib").exists(),
    reason="Cần features.parquet + trained forecast models.",
)


@pytest.fixture(scope="module")
def features_df():
    return pd.read_parquet(FEATURES_PATH)


# ---------- build_series ------------------------------------------------


def test_build_series_returns_tuple(features_df):
    target, exog = build_series(features_df, 956, "Superior City View")
    assert isinstance(target, pd.Series)
    assert isinstance(exog, pd.DataFrame)


def test_build_series_target_exog_aligned(features_df):
    target, exog = build_series(features_df, 956, "Superior City View")
    assert target.index.equals(exog.index)


def test_build_series_exog_columns(features_df):
    _, exog = build_series(features_df, 956, "Superior City View")
    assert list(exog.columns) == EXOG_COLS


def test_build_series_freq_daily(features_df):
    target, _ = build_series(features_df, 956, "Superior City View")
    assert target.index.freq is not None
    assert target.index.freqstr == "D"


def test_build_series_no_nan_with_default_tolerance(features_df):
    """Với target_lead_time=30, tolerance=5 → mỗi stay_date trong range
    đều có snapshot match → không NaN."""
    target, exog = build_series(
        features_df, 956, "Superior City View",
        target_lead_time=30, tolerance=5,
    )
    assert target.isna().sum() == 0
    assert exog.isna().sum().sum() == 0


def test_build_series_consistent_across_room_types(features_df):
    """Mỗi room_type có cùng số stay_dates valid (data structure đối xứng)."""
    lengths = [
        len(build_series(features_df, 956, rt, target_lead_time=30, tolerance=5)[0])
        for rt in [
            "Superior City View", "Deluxe City View Room",
            "Deluxe with banquette seating", "Premier city view",
            "Deluxe City View with banquette seating",
        ]
    ]
    assert len(set(lengths)) == 1, f"Series lengths khác nhau: {lengths}"


# ---------- ForecastModel predict ---------------------------------------


@pytest.fixture(scope="module")
def loaded_model():
    return ForecastModel.load(model_path(956, "Superior City View"))


@pytest.fixture(scope="module")
def future_exog(features_df, loaded_model):
    """Exog cho 7 ngày sau train end — dùng calendar cycling từ panel."""
    target, exog = build_series(features_df, 956, "Superior City View")
    return exog.iloc[-7:]


def test_predict_returns_forecast_result(loaded_model, future_exog):
    fc = loaded_model.predict(n_periods=7, exog_future=future_exog,
                                index=future_exog.index)
    assert isinstance(fc, ForecastResult)


def test_predict_shape(loaded_model, future_exog):
    fc = loaded_model.predict(n_periods=7, exog_future=future_exog,
                                index=future_exog.index)
    assert len(fc.p10) == 7
    assert len(fc.p50) == 7
    assert len(fc.p90) == 7


def test_predict_ci_ordered(loaded_model, future_exog):
    """p10 ≤ p50 ≤ p90 ở mọi điểm."""
    fc = loaded_model.predict(n_periods=7, exog_future=future_exog,
                                index=future_exog.index)
    assert (fc.p10 <= fc.p50).all()
    assert (fc.p50 <= fc.p90).all()


def test_predict_index_matches_input(loaded_model, future_exog):
    fc = loaded_model.predict(n_periods=7, exog_future=future_exog,
                                index=future_exog.index)
    assert fc.index.equals(future_exog.index)


def test_to_frame_columns(loaded_model, future_exog):
    fc = loaded_model.predict(n_periods=3, exog_future=future_exog.iloc[:3],
                                index=future_exog.index[:3])
    df = fc.to_frame()
    assert list(df.columns) == ["p10", "p50", "p90"]
    assert df.shape == (3, 3)


def test_unfit_model_predict_raises():
    """Predict trước fit → RuntimeError."""
    m = ForecastModel()
    with pytest.raises(RuntimeError, match="chưa fit"):
        m.predict(n_periods=1)


# ---------- Save / load roundtrip ---------------------------------------


def test_save_load_roundtrip(loaded_model, future_exog):
    """Predict trước vs sau save→load phải identical."""
    fc_before = loaded_model.predict(n_periods=3, exog_future=future_exog.iloc[:3],
                                       index=future_exog.index[:3])
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp = Path(f.name)
    try:
        loaded_model.save(tmp)
        roundtrip = ForecastModel.load(tmp)
        fc_after = roundtrip.predict(n_periods=3, exog_future=future_exog.iloc[:3],
                                       index=future_exog.index[:3])
        np.testing.assert_allclose(fc_before.p50.values, fc_after.p50.values)
        assert roundtrip.hotel_id == loaded_model.hotel_id
        assert roundtrip.room_type == loaded_model.room_type
    finally:
        tmp.unlink(missing_ok=True)
