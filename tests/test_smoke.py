"""Smoke tests — package import + helper sanity. Sẽ mở rộng ở Bước 7."""
from __future__ import annotations


def test_import_package():
    import src
    assert src.__version__


def test_import_modules():
    from src import data, features, forecast, demand, pricing
    assert hasattr(data, "load_raw")
    assert hasattr(features, "build_features")
    assert hasattr(forecast, "ForecastModel")
    assert hasattr(demand, "DemandModel")
    assert hasattr(pricing, "optimize_price")
