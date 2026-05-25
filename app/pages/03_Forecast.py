"""Page: Bước 3 — SARIMAX forecast (rendered notebook HTML)."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.title("Bước 3 — SARIMAX Forecast")
st.caption("Notebook `03_forecast.ipynb` — train 5 models per (hotel, room_type), MAPE/Coverage metrics.")

html_path = Path(__file__).resolve().parents[2] / "notebooks" / "03_forecast.html"
if html_path.exists():
    components.html(html_path.read_text(), height=4000, scrolling=True)
else:
    st.error(f"Không tìm thấy {html_path}.")
