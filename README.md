# Dynamic Pricing

POC dự đoán giá phòng và xác suất booking cho chuỗi hotel M Village / SAVVY.

Pipeline:

```
SAVVY-2BT.csv  ─►  src/data.py     ─►  src/features.py  ─►  data/processed/features.parquet
                                                              │
                            ┌─────────────────────────────────┴─────────────────┐
                            ▼                                                   ▼
                    src/forecast.py                                      src/demand.py
                    (SARIMAX per series)                                 (LogReg / LightGBM)
                            │                                                   │
                            └─────────────────► src/pricing.py ◄────────────────┘
                                                expected_revenue(X) = X × P(book | X)
                                                            │
                                                            ▼
                                                  app/streamlit_app.py
```

## Cấu trúc thư mục

```
dynamic-pricing/
├── ROADMAP.md                      # kế hoạch chi tiết, các quyết định kỹ thuật
├── README.md                       # (file này)
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/SAVVY-2BT.csv           # symlink, không commit
│   └── processed/features.parquet  # gen ra từ src/features.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_forecast.ipynb
│   └── 04_demand_curve.ipynb
├── src/                            # production code
│   ├── data.py                     # load + clean
│   ├── features.py                 # feature engineering
│   ├── forecast.py                 # SARIMAX forecast model
│   ├── demand.py                   # demand classifier
│   └── pricing.py                  # optimize price (argmax expected revenue)
├── models/                         # joblib artifacts (gitignored)
├── app/streamlit_app.py            # UI cho Sale team
└── tests/                          # pytest
```

## Setup

```bash
cd dynamic-pricing
python3.11 -m venv .venv          # Python 3.11/3.12 ổn nhất với pmdarima
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy app

```bash
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

App chạy ở `http://localhost:8501`.

## Workflow phát triển

Đi tuần tự theo `ROADMAP.md`:

1. **Bước 1 — EDA** → `notebooks/01_eda.ipynb`
2. **Bước 2 — Features** → fill `src/features.py`, sinh `data/processed/features.parquet`
3. **Bước 3 — Forecast** → fill `src/forecast.py`, train + save `models/forecast_*.joblib`
4. **Bước 4 — Demand** → fill `src/demand.py`, train + save `models/demand.joblib`
5. **Bước 5 — Pricing** → fill `src/pricing.py` + unit test
6. **Bước 6 — App** → fill `app/streamlit_app.py`
7. **Bước 7 — Polish** → tests, doc, deploy notes

## Data

`data/raw/SAVVY-2BT.csv` (gitignored, ~7MB) — snapshot inventory M Village SAVVY HBT.
Symlink từ `../../SAVVY-2BT.csv` (workspace root). Schema xem `ROADMAP.md` §1.

## Tech stack (đã chốt — xem `ROADMAP.md` §5)

- Forecast: **SARIMAX** (`pmdarima.auto_arima`), per `(hotel_id, room_type)`, weekly seasonality `m=7`, 80% CI native.
- Demand: **Logistic Regression baseline → LightGBM classifier**, target binary `did_book`.
- Holiday: **`holidays.country_holidays('VN')`**.
- App: **Streamlit + Plotly**.
