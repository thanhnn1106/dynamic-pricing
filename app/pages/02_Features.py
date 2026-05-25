"""Page: Bước 2 — Feature engineering (rendered notebook HTML)."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Features — Bước 2", layout="wide")
st.title("Bước 2 — Feature engineering")
st.caption("Notebook `02_features.ipynb` — lead_time bucket, calendar, cyclic encoding, did_book label.")

html_path = Path(__file__).resolve().parents[2] / "notebooks" / "02_features.html"
if html_path.exists():
    components.html(html_path.read_text(), height=4000, scrolling=True)
else:
    st.error(f"Không tìm thấy {html_path}.")
