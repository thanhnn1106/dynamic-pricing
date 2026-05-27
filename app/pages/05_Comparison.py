"""Page: Bước 8 (extra) — Forecast model comparison (rendered notebook HTML)."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.title("Bước 8 (extra) — Forecast Model Comparison")
st.caption("Notebook `05_model_comparison.ipynb` — SARIMAX vs Prophet trên 5 series.")

html_path = Path(__file__).resolve().parents[2] / "notebooks" / "05_model_comparison.html"
if html_path.exists():
    components.html(html_path.read_text(), height=4000, scrolling=True)
else:
    st.error(f"Không tìm thấy {html_path}.")
