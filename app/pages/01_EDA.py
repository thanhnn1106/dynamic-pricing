"""Page: Bước 1 — EDA (rendered notebook HTML)."""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.title("Bước 1 — EDA")
st.caption("Notebook `01_eda.ipynb` — schema check, cardinality, price variation, SARIMA feasibility.")

html_path = Path(__file__).resolve().parents[2] / "notebooks" / "01_eda.html"
if html_path.exists():
    components.html(html_path.read_text(), height=4000, scrolling=True)
else:
    st.error(f"Không tìm thấy {html_path}. Chạy `jupyter nbconvert --to html notebooks/01_eda.ipynb`.")
