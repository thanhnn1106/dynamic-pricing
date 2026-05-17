# Dynamic Pricing — Kế hoạch dự án

## 0. Mục tiêu

Xây hai model + một internal app cho Sale team:

- **Forecast model** — dự báo giá phòng tương lai theo `(hotel, room_type, stay_date)`, kèm confidence interval (CI 80%).
- **Demand model** — với mỗi mức giá ứng viên, dự đoán `P(book)`. Đây là demand curve, dùng để tìm giá tối ưu doanh thu.
- **Streamlit app** — flow:
  1. Sale chọn chi nhánh + loại phòng + date range
  2. Click **Get Price** → hiện đồ thị forecast (median line + CI band)
  3. Click **Get Dynamic Pricing** → hiện đồ thị `price` vs `P(book)` và `price` vs `expected_revenue`, đề xuất giá tối ưu

## 1. Hiểu dữ liệu — `SAVVY-2BT.csv`

55,225 dòng. Các cột quan trọng:

| Cột | Ý nghĩa |
|---|---|
| `updated_date` | Ngày chụp snapshot |
| `date` | Stay night (đêm khách thực ở) |
| `hotel_id`, `hotel_name` | Chi nhánh |
| `room_type_name`, `room_type_segment` | Loại phòng |
| `total`, `total_booked`, `total_maintenance`, `available` | Inventory |
| `price`, `ota_price` | Giá direct & giá OTA |

**Feature quan trọng nhất cần derive: `lead_time = date − updated_date`** (số ngày trước stay night). Trong hotel pricing, lead_time là biến giải thích price/demand mạnh nhất.

⚠️ Verify ở Bước 1: CSV này có bao nhiêu hotel? Sample chỉ thấy `hotel_id=956 (SAVVY HBT)`. Nếu chỉ 1 hotel → dropdown chi nhánh chỉ có 1 lựa chọn cho POC; sau này hook vào reporting Postgres để mở rộng.

## 2. Cấu trúc thư mục đề xuất

```
dynamic-pricing/
├── README.md
├── ROADMAP.md                    ← file này
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/SAVVY-2BT.csv         (gitignore — link/copy từ workspace)
│   └── processed/features.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_forecast.ipynb
│   └── 04_demand_curve.ipynb
├── src/                          (importable Python package)
│   ├── __init__.py
│   ├── data.py                   # load_csv, clean, train/val split
│   ├── features.py               # build_features()
│   ├── forecast.py               # ForecastModel: fit, predict + CI
│   ├── demand.py                 # DemandModel: P(book | price, features)
│   └── pricing.py                # optimize_price() = argmax(price × P(book))
├── models/                       (gitignore — joblib artifacts)
└── app/
    └── streamlit_app.py
```

## 3. Step-by-step

### Bước 0 — Setup (~30 phút)

- `python3 -m venv .venv && source .venv/bin/activate`
- Tạo `requirements.txt`: pandas, numpy, scikit-learn, **statsmodels**, **pmdarima**, lightgbm (cho demand model), plotly, streamlit, joblib, pyarrow, **holidays**, python-dateutil
- Mở rộng `.gitignore`: `.venv/`, `data/raw/*.csv`, `models/*.joblib`, `__pycache__/`, `.DS_Store`, `notebooks/.ipynb_checkpoints/`
- Tạo skeleton folders + empty `__init__.py`
- Symlink CSV: `ln -s ../../../../SAVVY-2BT.csv data/raw/SAVVY-2BT.csv` (từ `data/raw/` lên `AI-Course/` cần 4 cấp `../`)
- Commit `init: project scaffold` và push lên `github.com/thanhnn1106/dynamic-pricing`

### Bước 1 — EDA (~2–3 giờ) → `notebooks/01_eda.ipynb`

- Load CSV, check dtypes, missing, duplicates
- Đếm số hotel, room_type; range (`updated_date`, `date`) min/max
- Histogram `lead_time` (sau khi derive)
- Plot `price` theo `date`, theo `room_type` (line chart)
- Plot `occupancy_pct = total_booked/total` theo `date`
- Scatter `price` vs `lead_time`, `price` vs `occupancy_pct`, `price` vs `day_of_week`
- Correlation matrix với numerical features
- Heatmap (day_of_week, lead_time_bucket) → mean price
- **Sanity check biến động giá**: trong cùng `(hotel, room_type, stay_date)`, price có thay đổi theo `updated_date` không? Nếu giá gần như cố định → demand model khó học elasticity. Đây là quyết định kỹ thuật ở Bước 4.

**Deliverable**: bullet list 5–8 insight để chọn features ở Bước 2.

### Bước 2 — Feature engineering (~2 giờ) → `src/features.py`

```python
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Lead time
    df['lead_time_days'] = (df['date'] - df['updated_date']).dt.days
    df['lead_time_bucket'] = pd.cut(df['lead_time_days'],
        bins=[-1, 1, 7, 30, 90, 365], labels=['same', '<1w', '<1m', '<3m', '>3m'])

    # Calendar
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    # Vietnam public holidays via `holidays` package (Tết, 30/4, 2/9, ...)
    import holidays
    vn = holidays.country_holidays('VN', years=range(2024, 2028))
    df['is_holiday'] = df['date'].dt.date.isin(vn).astype(int)

    # Inventory state
    df['occupancy_pct'] = df['total_booked'] / df['total']
    df['available_pct'] = df['available'] / df['total']

    # Lag / rolling — same (hotel, room_type, dow), n days before
    df = df.sort_values(['hotel_id', 'room_type_name', 'date', 'updated_date'])
    df['price_lag7']     = df.groupby([...])['price'].shift(7)
    df['price_roll14']   = df.groupby([...])['price'].rolling(14).mean()
    return df
```

**Output**: `data/processed/features.parquet`

### Bước 3 — Forecast model (~1 ngày) → `src/forecast.py`

**Approach: SARIMA(X) — Seasonal ARIMA với exogenous features.**

**Cách tổ chức dữ liệu cho SARIMA**:

SARIMA cần *một time series liên tục theo ngày*, không phải bảng panel nhiều snapshot. Vì mỗi `(hotel, room_type, stay_date)` có nhiều `updated_date` snapshot, ta phải chọn một giá đại diện cho mỗi stay_date:

```python
# Cho mỗi (hotel, room_type, stay_date), lấy price từ snapshot mới nhất
# (giá "chốt" gần stay date nhất)
ts = (df.sort_values('updated_date')
        .groupby(['hotel_id', 'room_type_name', 'date'])
        .tail(1))
# → ts['date'] là stay_date, ts['price'] là target series
```

Fit **một SARIMA model riêng cho mỗi `(hotel, room_type)`** — series độc lập.

**Model**:

```python
from pmdarima import auto_arima

class ForecastModel:
    def fit(self, series: pd.Series, exog: pd.DataFrame = None):
        # series: index = stay_date (daily), values = price
        # exog: same index, columns = is_holiday, is_weekend, ...
        self.model = auto_arima(
            series,
            X=exog,
            seasonal=True, m=7,          # weekly seasonality
            d=None, D=None,              # auto-detect differencing
            stepwise=True,               # faster than full grid
            suppress_warnings=True,
            error_action='ignore',
            max_p=3, max_q=3, max_P=2, max_Q=2,
        )

    def predict(self, n_periods: int, exog_future: pd.DataFrame = None):
        forecast, conf_int = self.model.predict(
            n_periods=n_periods, X=exog_future,
            return_conf_int=True, alpha=0.2,   # 80% CI
        )
        return pd.DataFrame({
            'p10': conf_int[:, 0],
            'p50': forecast,
            'p90': conf_int[:, 1],
        })
```

- **Target**: `price` (giá direct, không phải `ota_price`)
- **Exogenous features (X)**: `is_holiday`, `is_weekend`, `dow` (sin/cos encoding), `month` (sin/cos)
  - KHÔNG dùng `occupancy`, `available` ở đây vì giá trị tương lai unknown → không feed làm exog của forecast được
  - Lead-time / lag features cũng skip — SARIMA tự handle autoregressive lag
- **Index**: stay_date làm DatetimeIndex tần suất `D` (daily); reindex để không có gap
- **Split**: time-based — train trên đầu series, hold out 30 ngày cuối làm validation
- **Metric**: MAPE + RMSE trên hold-out; coverage = % thực tế nằm trong [p10, p90] (target 80%)
- **Save**: pickle `models/forecast_{hotel_id}_{room_type_slug}.joblib`

**Pros/Cons** của SARIMA so với ML approach:

| Pros | Cons |
|---|---|
| Native CI (no quantile loss cần) | Một model/series, không share info giữa các room_type |
| Decompose được trend + seasonality để Sale hiểu | Cần series đủ dài (≥60-90 ngày) mới ổn định |
| `auto_arima` tự pick (p,d,q)(P,D,Q) | Khó dùng nhiều exog features (có nhưng giới hạn) |
| Statistical assumption rõ ràng | Sensitive với outlier / missing dates |

⚠️ **Verify ở B1**: series có đủ dài không? Sample CSV bắt đầu `updated_date=2026-05-04`. Nếu data ngắn hơn 60 ngày cho mỗi (hotel, room_type) → SARIMA sẽ underfit; có thể phải dùng SES/Holt-Winters đơn giản hơn hoặc gộp lại theo room_type_segment.

### Bước 4 — Demand model (~1 ngày) → `src/demand.py`

**Câu hỏi**: với giá X tại `(hotel, room_type, stay_date, lead_time, occupancy)`, xác suất có booking xảy ra trong các ngày tới là bao nhiêu?

**Cách build target** từ snapshots:

```python
# Pair các snapshot liên tiếp của cùng (hotel, room_type, stay_date)
df = df.sort_values(['hotel_id', 'room_type_name', 'date', 'updated_date'])
df['next_total_booked'] = df.groupby(['hotel_id', 'room_type_name', 'date'])['total_booked'].shift(-1)
df['delta_booked'] = df['next_total_booked'] - df['total_booked']
df['did_book'] = (df['delta_booked'] > 0).astype(int)
```

**Model**:
- Baseline: Logistic Regression (interpretable, có signed coefficient cho price → elasticity)
- Production: LGBMClassifier với `predict_proba`
- **Features**: `price` (KEY), `occupancy_pct`, `lead_time_days`, `dow`, `is_weekend`, `is_holiday`, `room_type_segment`, `available`

**⚠️ Caveat lớn**: dữ liệu chỉ có giá thực tế đã set — không có counterfactual ("nếu set giá khác thì sao"). Model học correlation chứ chưa hẳn causal. Nếu Bước 1 phát hiện giá biến động ít trong cùng context → đánh giá elasticity weak; ghi rõ trong README rằng đây là *predictive* demand chứ chưa phải *causal* price elasticity. Phase 2 cần A/B test thật để hiệu chỉnh.

### Bước 5 — Pricing optimization (~4 giờ) → `src/pricing.py`

```python
def optimize_price(features_row, forecast_model, demand_model,
                   price_grid=None) -> dict:
    """Cho mỗi giá trên grid, tính expected_revenue = price × P(book).
    Trả về optimal_price, max_revenue, full curves."""
    if price_grid is None:
        f = forecast_model.predict(features_row)
        price_grid = np.linspace(f['p10'] * 0.7, f['p90'] * 1.3, 80)

    probs = []
    for p in price_grid:
        x = features_row.copy()
        x['price'] = p
        probs.append(demand_model.predict_proba(x)[0, 1])
    probs = np.array(probs)
    expected_rev = price_grid * probs
    i = expected_rev.argmax()
    return {
        'price_grid': price_grid,
        'prob_book': probs,
        'expected_revenue': expected_rev,
        'optimal_price': price_grid[i],
        'optimal_prob': probs[i],
        'max_revenue': expected_rev[i],
    }
```

### Bước 6 — Streamlit app (~1 ngày) → `app/streamlit_app.py`

**Layout**:

```
┌─ Sidebar ─────────────┐  ┌─ Main ───────────────────────────────────┐
│ Chi nhánh: [selectbox]│  │ [Get Price Forecast]                     │
│ Loại phòng: [selectbox]│  │ ┌─────────────────────────────────────┐  │
│ Từ ngày: [date]       │  │ │ Plotly: median + CI band             │  │
│ Đến ngày: [date]      │  │ └─────────────────────────────────────┘  │
└───────────────────────┘  │ [Get Dynamic Pricing]                   │
                            │ ┌────────────────┐  ┌─────────────────┐ │
                            │ │ P(book) vs $   │  │ ExpRev vs $    │ │
                            │ └────────────────┘  └─────────────────┘ │
                            │ Bảng: date | forecast | optimal | rev   │
                            └──────────────────────────────────────────┘
```

- `@st.cache_data` cho `load_features()`
- `@st.cache_resource` cho `load_forecast_model()` / `load_demand_model()`
- State giữa 2 button: `st.session_state['forecast_done'] = True`

### Bước 7 — Polish (~4 giờ)

- `README.md`: setup, how-to-run, sample screenshot
- Pin versions trong `requirements.txt`
- Unit test cho `src/data.py` (schema), `src/forecast.py` (predict shape), `src/pricing.py` (optimal nằm trong grid)
- Logging cơ bản
- (Sau) Dockerfile để Sale team không cần cài Python

## 4. Timeline

| # | Bước | Thời gian | Deliverable |
|---|---|---|---|
| 0 | Setup | 30m | git push đầu tiên |
| 1 | EDA | 2–3h | notebook + insights list |
| 2 | Features | 2h | `features.parquet` + `src/features.py` |
| 3 | Forecast | 1d | model artifact + eval plot |
| 4 | Demand | 1d | model artifact + elasticity report |
| 5 | Pricing | 4h | `src/pricing.py` + unit test |
| 6 | App | 1d | Streamlit chạy được |
| 7 | Polish | 4h | README + tests |

**Tổng: ~5–6 ngày full-time**, hoặc spread ~2 tuần part-time.

## 5. Quyết định đã chốt

1. **Forecast**: SARIMA(X) — `pmdarima.auto_arima` với weekly seasonality `m=7`, exog = holiday/weekend/dow/month, 80% CI native.
2. **Demand target**: binary `did_book` — classifier trả `P(book)`.
3. **Holiday**: package `holidays` với locale VN (`holidays.country_holidays('VN')`).
4. **Granularity**: per-day-per-`(hotel, room_type)`.

Sẵn sàng vào Bước 0 khi bạn OK.
