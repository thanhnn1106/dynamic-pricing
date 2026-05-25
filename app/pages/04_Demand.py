"""Page: Bước 4 — Demand model + curve (rendered notebook HTML)."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.title("Bước 4 — Demand Model")
st.caption("Notebook `04_demand_curve.ipynb` — LogReg + LightGBM, elasticity check, demand curve viz.")

html_path = Path(__file__).resolve().parents[2] / "notebooks" / "04_demand_curve.html"
if html_path.exists():
    components.html(html_path.read_text(), height=4000, scrolling=True)
else:
    st.error(f"Không tìm thấy {html_path}.")
