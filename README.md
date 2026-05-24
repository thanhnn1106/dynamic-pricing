# Dynamic Pricing

POC dự đoán giá phòng và xác suất booking cho chuỗi khách sạn (dataset đã sanitize hotel_name cho public POC).

Pipeline:

```
SAMV-HBT.csv  ─►  src/data.py     ─►  src/features.py  ─►  data/processed/features.parquet
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

## Prerequisites

- **Python 3.11 hoặc 3.12** — `pmdarima` chưa stable trên 3.13+
- **macOS**: cần `libomp` cho LightGBM
  ```bash
  brew install libomp
  ```
- **Linux**: thường đã có `libgomp` qua build-essential

## Setup

```bash
cd dynamic-pricing
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Symlink data:
```bash
# Từ data/raw/, link CSV (path tuỳ workspace của bạn)
ln -s ../../../SAMV-HBT.csv data/raw/SAMV-HBT.csv
```

## Quick start — chạy end-to-end

```bash
source .venv/bin/activate

# 1. Sinh features (clean + feature engineering, ~5s)
python -c "from src.features import prepare_features; prepare_features()"

# 2. Train 5 SARIMAX models (~2-3 phút)
python -c "from src.forecast import train_all; print(train_all())"

# 3. Train LogReg + LightGBM demand models (~10s)
python -c "from src.demand import train_all; print(train_all())"

# 4. Chạy app
streamlit run app/streamlit_app.py
```

App tại `http://localhost:8501`.

## App usage (cho Sale team)

1. **Sidebar**: chọn chi nhánh + loại phòng + date range (max 60 ngày sau train end)
2. **Get Price Forecast** → đồ thị median + 80% CI band cho range
3. **Get Dynamic Pricing** → bảng per-date với:
   - `forecast_p50` — giá đoán
   - `optimal_price` — giá đề xuất tối đa hoá expected revenue
   - `optimal_P(book)` — xác suất book ở giá đó
   - `max_revenue` — `optimal_price × P(book)`
   - Cờ `⚠️ at grid edge` (elasticity yếu, optimal ngoài range explore)
   - Cờ `🟡 CI crossed 0` (forecast unreliable, p10 hoặc p90 âm)
4. **Drill-down**: chọn 1 ngày → 2 chart `P(book) vs price` + `expected_revenue vs price`

## Tests

```bash
pytest tests/ -v
```

38 tests covering: data schema, clean logic, build_series, ForecastModel save/load, optimize_price invariants (optimal ∈ grid, max_revenue = argmax), edge cases (p10/p90 âm, p50 ≤ 0).

## Deploy lên Streamlit Cloud

App đã được setup để deploy thẳng lên [share.streamlit.io](https://share.streamlit.io) (free tier).

**File đã commit cho deploy** (override `.gitignore`):
- `data/processed/features.parquet` (104KB) — input cho app
- `models/forecast_*.joblib` × 5 (9.4MB) + `models/demand_*.joblib` × 2 (1.3MB)
- `.python-version` → 3.11 (Streamlit Cloud auto-detect)
- `packages.txt` → `libgomp1` (LightGBM Linux dependency)
- `.streamlit/config.toml` → theme + headless mode

**KHÔNG commit**:
- `data/raw/*.csv` — chỉ cần khi re-train, app runtime đọc parquet
- `.streamlit/secrets.toml` — credentials (nếu có sau này)

### Steps để deploy

1. **Push repo lên GitHub** (public):
   ```bash
   git push -u origin sanitize-hotel-name      # hoặc merge → main rồi push main
   ```

2. **Vào [share.streamlit.io](https://share.streamlit.io)** → "New app":
   - Repository: `<your-username>/dynamic-pricing`
   - Branch: branch chứa deploy artifacts (vd `main` hoặc `sanitize-hotel-name`)
   - Main file path: `app/streamlit_app.py`
   - Python version: 3.11 (auto-detect từ `.python-version`)

3. **Click Deploy**. First boot ~2 phút (pip install pmdarima + lightgbm). Subsequent boot ~30s (cached).

4. App URL sẽ là `https://<your-slug>.streamlit.app`.

### Re-deploy

Streamlit Cloud auto-redeploy khi push lên branch đã connect. Nếu re-train models locally rồi push, app tự reload với models mới.

### Limitations free tier

- 1GB memory — đủ cho 7 joblibs + parquet
- App "ngủ" sau ~1 tuần không có traffic, wake-up khi có request mới (lag 30s)
- Không persistent storage — mỗi reboot mất state

## Workflow phát triển (đã xong cả 7 bước)

| # | Bước | Output |
|---|---|---|
| 1 | EDA | `notebooks/01_eda.ipynb` — 5 insight key, filter post-stay |
| 2 | Features | `src/features.py` + `data/processed/features.parquet` (46,480 × 27) |
| 3 | Forecast | `src/forecast.py` + 5 `models/forecast_*.joblib`, MAPE 20-31%, Coverage 73% |
| 4 | Demand | `src/demand.py` + 2 `models/demand_*.joblib`, AUC 0.88 |
| 5 | Pricing | `src/pricing.py` + 14 unit tests |
| 6 | App | `app/streamlit_app.py` — Sale UI |
| 7 | Polish | README + 38 tests + version pinning |

Chi tiết kế hoạch + quyết định kỹ thuật xem `ROADMAP.md`.

## Cấu trúc thư mục

```
dynamic-pricing/
├── ROADMAP.md                      # kế hoạch chi tiết
├── README.md                       # (file này)
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/SAMV-HBT.csv           # symlink, gitignored
│   └── processed/features.parquet  # gen ra từ prepare_features()
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_forecast.ipynb
│   └── 04_demand_curve.ipynb
├── src/                            # production code
│   ├── data.py                     # load_raw, clean
│   ├── features.py                 # build_features, prepare_features, save_features
│   ├── forecast.py                 # ForecastModel, build_series, train_all
│   ├── demand.py                   # DemandModel, prepare_X_y, time_split, train_all
│   └── pricing.py                  # optimize_price, PricingCurve
├── models/                         # joblib artifacts (gitignored)
├── app/streamlit_app.py            # UI cho Sale team
└── tests/                          # pytest (38 tests)
```

## Tech stack (đã chốt — xem `ROADMAP.md` §5)

- Forecast: **SARIMAX** (`pmdarima.auto_arima`), per `(hotel_id, room_type)`, weekly seasonality `m=7`, 80% CI native.
- Demand: **Logistic Regression baseline + LightGBM**, target binary `did_book`.
- Holiday: **`holidays.country_holidays('VN')`**.
- App: **Streamlit + Plotly**.

## Known limitations (POC scope)

1. **Lệch ROADMAP**: forecast target dùng "snapshot ở fixed lead_time=30" thay vì "latest snapshot per stay_date". Lý do: `updated_date` data chỉ 2026-01-02 → 2026-05-04, late stay_dates không có near-stay snapshot → train/holdout khác domain → MAPE 66% với convention ROADMAP. Fixed lead_time → MAPE 27%. Chi tiết: commit message Bước 3.

2. **MAPE forecast 20-31%** vẫn cao cho production (mục tiêu <10%). Phase 2: log-transform price, data dài hơn (cần ≥6 tháng updated_date), thêm exog (event calendar).

3. **Demand elasticity yếu**: LogReg `price` coef chỉ -0.21 (mạnh: lead_time -1.73). Optimal price hay rơi vào edge grid → cờ `⚠️` trong app. Phase 2 cần A/B test thật để có causal estimate.

4. **Predictive ≠ causal**: data thiếu counterfactual. Model học correlation. Không khuyến nghị deploy autonomous mà cần Sale review.

5. **Forecast horizon ≤ 60 ngày**: xa hơn CI band phình to/âm, app cap cứng để tránh user nhận garbage output.

## Data

`data/raw/SAMV-HBT.csv` (gitignored, ~7MB) — snapshot inventory cho 1 hotel (`hotel_name = 'SAMV HBT'`, sanitized).

Schema (13 cột):

| Cột | Ý nghĩa |
|---|---|
| `updated_date` | Ngày chụp snapshot |
| `date` | Stay night (đêm khách thực ở) |
| `hotel_id`, `hotel_name` | Chi nhánh (POC chỉ có 1 hotel) |
| `room_type_name`, `room_type_segment`, `brand_sub_segment` | Phân loại phòng |
| `total`, `total_booked`, `total_maintenance`, `available` | Inventory state |
| `price`, `ota_price` | Giá direct & OTA |

55,225 rows raw → 46,480 sau `clean()` (drop 15.84% post-stay snapshots).
