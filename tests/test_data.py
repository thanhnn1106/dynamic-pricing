"""Tests cho `src.data` — schema + clean logic."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import clean, load_raw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "SAVVY-2BT.csv"

pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists(),
    reason="Cần data/raw/SAVVY-2BT.csv (symlink workspace CSV).",
)


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def clean_df(raw_df):
    return clean(raw_df)


# ---------- Schema -------------------------------------------------------


def test_load_raw_shape(raw_df):
    """55,225 dòng × 13 cột — đã confirm ở EDA Bước 1."""
    assert raw_df.shape == (55225, 13)


EXPECTED_COLS = {
    "updated_date", "date", "hotel_id", "hotel_name", "room_type_name",
    "total_booked", "total_maintenance", "total", "available",
    "price", "ota_price", "room_type_segment", "brand_sub_segment",
}


def test_load_raw_columns(raw_df):
    assert set(raw_df.columns) == EXPECTED_COLS


def test_dates_parsed_as_datetime(raw_df):
    assert pd.api.types.is_datetime64_any_dtype(raw_df["updated_date"])
    assert pd.api.types.is_datetime64_any_dtype(raw_df["date"])


def test_numeric_cols_are_numeric(raw_df):
    for col in ("total_booked", "total_maintenance", "total", "available",
                "price", "ota_price", "hotel_id"):
        assert pd.api.types.is_numeric_dtype(raw_df[col]), f"{col} not numeric"


# ---------- Clean --------------------------------------------------------


def test_clean_drops_post_stay(raw_df, clean_df):
    """clean() drop rows có updated_date > date (post-stay snapshot)."""
    dropped = len(raw_df) - len(clean_df)
    pct = dropped / len(raw_df)
    assert dropped == 8745         # số đã confirm ở Bước 1
    assert 0.15 < pct < 0.17       # ~15.84%


def test_clean_lead_time_nonneg(clean_df):
    """Sau clean: mọi row có date >= updated_date."""
    lead = (clean_df["date"] - clean_df["updated_date"]).dt.days
    assert (lead >= 0).all()


def test_clean_preserves_columns(raw_df, clean_df):
    assert set(clean_df.columns) == set(raw_df.columns)


def test_clean_idempotent(clean_df):
    """clean(clean(df)) == clean(df) — không drop thêm gì lần 2."""
    twice = clean(clean_df)
    assert len(twice) == len(clean_df)


def test_clean_no_duplicates(clean_df):
    """clean() đã drop_duplicates → không còn full-row duplicate."""
    assert clean_df.duplicated().sum() == 0
