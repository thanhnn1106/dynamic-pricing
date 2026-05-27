"""Streamlit app — Dynamic Pricing cho Sale team.

Flow:
    1. Sidebar: chọn chi nhánh + loại phòng + date range
    2. "Get Price Forecast" → Plotly: median + CI band cho range đã chọn
    3. "Get Dynamic Pricing" → table per-date + drill-down demand curve

Chạy:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.demand import DemandModel, MODELS_DIR
from src.forecast import (
    DEFAULT_FEATURES, EXOG_COLS, ForecastModel, ForecastResult, model_path,
)
from src.forecast_prophet import ProphetForecastModel, model_path_prophet
from src.forecast_lstm import LSTMForecastModel, model_path_lstm
from src.pricing import optimize_price


# ======================================================================
# Cached loaders
# ======================================================================
@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(DEFAULT_FEATURES)


@st.cache_resource
def load_forecast(hotel_id: int, room_type: str, kind: str = "sarimax"):
    """Dispatch 3-way. Tất cả mirror ForecastModel API.

    - `sarimax`: MAPE ~27%, CI ~73% (calibrated) — best cho pricing optimization
    - `prophet`: MAPE ~9%, CI ~25% (hẹp) — best point forecast theo decomposition
    - `lstm`: MAPE ~7%, CI ~4% (cực hẹp) — best point forecast (PyTorch + MC Dropout)
    """
    if kind == "prophet":
        return ProphetForecastModel.load(model_path_prophet(hotel_id, room_type))
    if kind == "lstm":
        return LSTMForecastModel.load(model_path_lstm(hotel_id, room_type))
    return ForecastModel.load(model_path(hotel_id, room_type))


@st.cache_resource
def load_demand(kind: str) -> DemandModel:
    return DemandModel.load(MODELS_DIR / f"demand_{kind}.joblib")


@st.cache_data
def vn_holidays() -> set:
    import holidays
    return set(holidays.country_holidays("VN", years=list(range(2024, 2028))).keys())


# ======================================================================
# Helpers
# ======================================================================
def calendar_exog(dates: pd.DatetimeIndex, holiday_set: set) -> pd.DataFrame:
    """Build EXOG_COLS cho 1 dải date — khớp schema train ở Bước 3."""
    dow = dates.dayofweek
    month = dates.month
    return pd.DataFrame({
        "is_holiday": pd.Series(dates.date).isin(holiday_set).astype(int).values,
        "is_weekend": (dow >= 5).astype(int),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    }, index=dates)[EXOG_COLS]


def forecast_range(model: ForecastModel, start: pd.Timestamp, end: pd.Timestamp,
                   holiday_set: set) -> ForecastResult:
    """Predict từ `last_train_date+1` đến `end`, slice về [start, end].

    SARIMAX rolling-forecast contiguous từ last_train_date — không thể "skip"
    đến date xa. Predict full range rồi mới slice.
    """
    first_future = model.last_train_date + pd.Timedelta(days=1)
    if end < first_future:
        raise ValueError(f"end={end.date()} trước first_future={first_future.date()}")
    full = pd.date_range(first_future, end, freq="D")
    exog = calendar_exog(full, holiday_set)
    fc = model.predict(n_periods=len(full), exog_future=exog, index=full)

    mask = (fc.index >= start) & (fc.index <= end)
    return ForecastResult(
        index=fc.index[mask],
        p10=fc.p10[mask], p50=fc.p50[mask], p90=fc.p90[mask],
    )


def build_context_row(stay_date: pd.Timestamp, hotel_id: int, room_type: str,
                       features_df: pd.DataFrame, holiday_set: set,
                       lead_time: int = 30) -> pd.Series:
    """Synthesize context row cho demand model.

    Future stay_date không có snapshot → dùng median occupancy của room đó
    trong train làm proxy. lead_time=30 khớp convention forecast (B3).
    """
    room_rows = features_df[features_df["room_type_name"] == room_type]
    segment = room_rows["room_type_segment"].iloc[0]
    default_occ = float(room_rows["occupancy_pct"].median())
    default_avail = float(room_rows["available_pct"].median())

    dow = stay_date.dayofweek
    month = stay_date.month

    return pd.Series({
        "hotel_id": hotel_id,
        "room_type_name": room_type,
        "room_type_segment": segment,
        "date": stay_date,
        "price": float(room_rows["price"].median()),     # placeholder
        "occupancy_pct": default_occ,
        "available_pct": default_avail,
        "lead_time_days": float(lead_time),
        "is_weekend": int(dow >= 5),
        "is_holiday": int(stay_date.date() in holiday_set),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
    })


def fmt_vnd(v: float) -> str:
    return f"{v:,.0f} ₫"


# ======================================================================
# UI — main pricing page (wrapped as function cho st.navigation)
# ======================================================================
st.set_page_config(page_title="Dynamic Pricing — SAMV HBT", layout="wide")


def pricing_page():
    st.title("Dynamic Pricing")
    st.caption("POC — forecast giá phòng + đề xuất giá tối ưu cho Sale team")

    features_df = load_features()
    holiday_set = vn_holidays()

    hotels = features_df[["hotel_id", "hotel_name"]].drop_duplicates().sort_values("hotel_id")
    room_types = sorted(features_df["room_type_name"].unique())

    with st.sidebar:
        st.header("Lựa chọn")
        hotel_name = st.selectbox("Chi nhánh", hotels["hotel_name"].tolist())
        hotel_id = int(hotels.loc[hotels["hotel_name"] == hotel_name, "hotel_id"].iloc[0])

        room_type = st.selectbox("Loại phòng", room_types)

        forecast_kind = st.radio(
            "Forecast model", ["sarimax", "prophet", "lstm"], horizontal=True,
            help="SARIMAX: MAPE 27%, CI 73% (best cho pricing). "
                 "Prophet: MAPE 9%, CI 25%. "
                 "LSTM: MAPE 7% (best), CI 4% (rất hẹp).",
        )
        if forecast_kind in ("prophet", "lstm"):
            st.caption(f"🟡 {forecast_kind.upper()} CI hẹp → pricing grid p50-anchor fallback.")

        # Last train date của model này — đảm bảo date_from > đó
        fmodel = load_forecast(hotel_id, room_type, kind=forecast_kind)
        min_date = (fmodel.last_train_date + pd.Timedelta(days=1)).date()
        # Cap forecast horizon: 60 ngày sau train end. Xa hơn → CI band
        # phình to/âm (chỉ 103 ngày train), unreliable cho pricing.
        max_date = (fmodel.last_train_date + pd.Timedelta(days=60)).date()
        default_from = max(dt.date.today() + dt.timedelta(days=1), min_date)
        if default_from > max_date:
            default_from = min_date
        date_from = st.date_input("Từ ngày", default_from,
                                    min_value=min_date, max_value=max_date)
        date_to = st.date_input("Đến ngày", min(default_from + dt.timedelta(days=6), max_date),
                                  min_value=date_from, max_value=max_date)
        st.caption(f"Forecast horizon: tối đa {max_date} (60d sau train end).")

        st.divider()
        demand_kind = st.radio("Demand model", ["lgbm", "logreg"], horizontal=True,
                                help="LGBM calibrate tốt hơn; LogReg interpretable")
        st.caption(f"Model train end: **{fmodel.last_train_date.date()}**")

    # ----------------------------------------------------------------------
    # Section 1 — Price Forecast
    # ----------------------------------------------------------------------
    st.subheader("1. Price Forecast")

    if st.button("Get Price Forecast", type="primary"):
        start = pd.Timestamp(date_from)
        end = pd.Timestamp(date_to)
        with st.spinner(f"Forecast {(end - start).days + 1} ngày..."):
            try:
                fc = forecast_range(fmodel, start, end, holiday_set)
            except ValueError as e:
                st.error(str(e))
                st.stop()
        st.session_state["fc"] = fc
        st.session_state["fc_context"] = (hotel_id, room_type, forecast_kind, demand_kind)

    if "fc" in st.session_state:
        fc: ForecastResult = st.session_state["fc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.p90, name="p90", mode="lines",
            line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.p10, name="80% CI", mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.2)",
        ))
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.p50, name="Forecast (p50)", mode="lines+markers",
            line=dict(color="rgb(99, 110, 250)", width=3),
        ))
        fig.update_layout(
            title=f"Forecast {room_type} — {fc.index.min().date()} → {fc.index.max().date()}",
            xaxis_title="Stay date", yaxis_title="Price (VND)",
            hovermode="x unified", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        fc_df = fc.to_frame().round(0).reset_index().rename(columns={"index": "date"})
        fc_df["dow"] = fc_df["date"].dt.day_name()
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------------------
    # Section 2 — Dynamic Pricing
    # ----------------------------------------------------------------------
    st.subheader("2. Dynamic Pricing")

    fc_ready = "fc" in st.session_state and st.session_state.get("fc_context") == (
        hotel_id, room_type, forecast_kind, demand_kind,
    )
    if not fc_ready:
        st.info("⏳ Click **Get Price Forecast** trước (cho cùng config).")

    if st.button("Get Dynamic Pricing", disabled=not fc_ready):
        fc = st.session_state["fc"]
        dmodel = load_demand(demand_kind)

        rows = []
        curves = {}
        skipped = []
        progress = st.progress(0.0, text="Optimizing...")
        for i, stay_date in enumerate(fc.index):
            ctx = build_context_row(stay_date, hotel_id, room_type, features_df, holiday_set)
            single_fc = ForecastResult(
                index=fc.index[i:i+1],
                p10=fc.p10.iloc[i:i+1], p50=fc.p50.iloc[i:i+1], p90=fc.p90.iloc[i:i+1],
            )
            try:
                curve = optimize_price(ctx, single_fc, dmodel)
            except ValueError as e:
                skipped.append((stay_date, str(e)))
                progress.progress((i + 1) / len(fc.index))
                continue
            curves[stay_date] = curve
            at_edge = (curve.optimal_price == curve.price_grid[-1]
                        or curve.optimal_price == curve.price_grid[0])
            # Cờ "forecast suspicious" — p10 hoặc p90 âm = CI band crossed zero
            # → forecast bản thân unreliable, dù optimizer đã p50-anchor.
            p10, p90 = fc.p10.iloc[i], fc.p90.iloc[i]
            forecast_warn = "🟡 CI crossed 0" if (p10 < 0 or p90 < 0) else ""
            rows.append({
                "date": stay_date,
                "dow": stay_date.day_name(),
                "forecast_p50": fc.p50.iloc[i],
                "optimal_price": curve.optimal_price,
                "optimal_P(book)": curve.optimal_prob,
                "max_revenue": curve.max_revenue,
                "edge_warning": "⚠️ at grid edge" if at_edge else "",
                "forecast_warn": forecast_warn,
            })
            progress.progress((i + 1) / len(fc.index))
        progress.empty()
        if skipped:
            st.error(f"{len(skipped)} ngày skip do forecast lỗi nặng:\n" +
                      "\n".join(f"  • {d.date()}: {msg}" for d, msg in skipped[:5]))

        st.session_state["pricing_rows"] = rows
        st.session_state["pricing_curves"] = curves

    if "pricing_rows" in st.session_state:
        rows = st.session_state["pricing_rows"]
        df_view = pd.DataFrame(rows)
        # Format display
        show = df_view.copy()
        for col in ("forecast_p50", "optimal_price", "max_revenue"):
            show[col] = show[col].apply(fmt_vnd)
        show["optimal_P(book)"] = show["optimal_P(book)"].round(3)
        st.dataframe(show, use_container_width=True, hide_index=True)

        n_edge = sum(1 for r in rows if r["edge_warning"])
        if n_edge:
            st.warning(
                f"{n_edge}/{len(rows)} ngày có optimal price ở edge grid → demand "
                "model elasticity yếu cho range price này. Nên đọc với caution."
            )

        # Drill-down
        st.markdown("### 🔍 Drill-down: demand curve cho 1 ngày")
        options = [r["date"] for r in rows]
        sel = st.selectbox("Chọn ngày", options, format_func=lambda d: d.strftime("%Y-%m-%d (%a)"))
        curve = st.session_state["pricing_curves"][sel]

        col1, col2 = st.columns(2)
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=curve.price_grid, y=curve.prob_book,
                                        mode="lines", name="P(book)"))
            fig1.add_vline(x=curve.optimal_price, line_dash="dash", line_color="red",
                            annotation_text=f"Optimal: {fmt_vnd(curve.optimal_price)}")
            fig1.update_layout(title="P(book) vs price", xaxis_title="Price (VND)",
                                yaxis_title="P(book)", height=380)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=curve.price_grid, y=curve.expected_revenue,
                                        mode="lines", name="Expected revenue"))
            fig2.add_vline(x=curve.optimal_price, line_dash="dash", line_color="red",
                            annotation_text=f"Max: {fmt_vnd(curve.max_revenue)}")
            fig2.update_layout(title="Expected revenue vs price",
                                xaxis_title="Price (VND)",
                                yaxis_title="Expected revenue (VND)", height=380)
            st.plotly_chart(fig2, use_container_width=True)


# ======================================================================
# Router — st.navigation (replace pages/ auto-discovery, more reliable
# trên Streamlit Cloud)
# ======================================================================
nav = st.navigation([
    st.Page(pricing_page, title="Dynamic Pricing", icon="💰", default=True),
    st.Page("pages/01_EDA.py", title="Bước 1 — EDA", icon="📊"),
    st.Page("pages/02_Features.py", title="Bước 2 — Features", icon="🛠️"),
    st.Page("pages/03_Forecast.py", title="Bước 3 — Forecast", icon="📈"),
    st.Page("pages/04_Demand.py", title="Bước 4 — Demand", icon="📉"),
    st.Page("pages/05_Comparison.py", title="Bước 8 — Compare", icon="⚖️"),
])
nav.run()
