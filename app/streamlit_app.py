"""Streamlit app — Dynamic Pricing cho Sale team.

Flow:
    1. Sidebar: chọn chi nhánh + loại phòng + date range
    2. Button "Get Price Forecast" → đồ thị median + CI band
    3. Button "Get Dynamic Pricing" → đồ thị P(book) vs price + expected_revenue vs price
       + bảng giá đề xuất

Skeleton hôm nay — UI hoàn thiện ở Bước 6.

Chạy:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import datetime as dt

import streamlit as st

st.set_page_config(page_title="Dynamic Pricing — M Village", layout="wide")

st.title("Dynamic Pricing")
st.caption("POC — forecast giá phòng + đề xuất giá tối ưu cho Sale team")

# ----------------------------------------------------------------------
# Sidebar — input
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Lựa chọn")
    # TODO Bước 6: load hotel list từ data/processed/features.parquet
    hotel = st.selectbox("Chi nhánh", ["SAVVY BY M VILLAGE HAI BÀ TRƯNG"])
    room_type = st.selectbox("Loại phòng", [
        "Superior City View",
        "Deluxe City View Room",
        "Deluxe with banquette seating",
        "Deluxe City View with banquette seating",
        "Premier city view",
    ])
    today = dt.date.today()
    date_from = st.date_input("Từ ngày", today + dt.timedelta(days=1))
    date_to = st.date_input("Đến ngày", today + dt.timedelta(days=7))

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.subheader("1. Price Forecast")
if st.button("Get Price Forecast", type="primary"):
    st.info("TODO Bước 6 — load src.forecast.ForecastModel, predict, plot Plotly.")
    st.session_state["forecast_done"] = True

st.subheader("2. Dynamic Pricing")
disabled = not st.session_state.get("forecast_done", False)
if st.button("Get Dynamic Pricing", disabled=disabled):
    st.info("TODO Bước 6 — gọi src.pricing.optimize_price, plot demand curve.")
