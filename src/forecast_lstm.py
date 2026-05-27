"""LSTM forecast model (PyTorch) — alternative cho SARIMAX + Prophet.

Architecture: 1-layer LSTM (hidden_size=32) + Dropout(0.3) + Dense(1).
Cố ý SMALL vì 103 obs train rất ít — net to sẽ overfit nặng.

CI estimation: MC Dropout — turn dropout ON tại inference, parallel
batch K trajectories với independent dropout masks → quantiles.

Iterative multi-step: predict 1 ngày, append prediction vào sequence,
predict tiếp. Errors compound nhưng standard approach.

Features per timestep (7 dim): price + EXOG_COLS (is_holiday, is_weekend,
dow_sin, dow_cos, month_sin, month_cos).

Note: KHÔNG dùng TensorFlow vì không cài được trên Mac M1 Pro x86_64
Rosetta venv (AVX requirement). PyTorch CPU wheels work fine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .forecast import EXOG_COLS, ForecastResult, _slugify


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch not installed. LSTM backend unavailable. "
            "Install với `pip install torch`."
        )

DEFAULT_FEATURES = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.parquet"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


if HAS_TORCH:
    class _LSTMNet(nn.Module):
        def __init__(self, feature_dim: int, hidden_size: int = 32,
                     num_layers: int = 1, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=feature_dim, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            last = self.dropout(last)
            return self.fc(last).squeeze(-1)
else:
    _LSTMNet = None    # placeholder — fit/predict raise via _require_torch()


class LSTMForecastModel:
    """PyTorch LSTM wrapper. Mirror ForecastModel API."""

    def __init__(self, seq_len: int = 14, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.3):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_dim: Optional[int] = None
        self.model: Optional[_LSTMNet] = None
        self.scaler_mean: Optional[np.ndarray] = None     # per-feature
        self.scaler_std: Optional[np.ndarray] = None
        self.hotel_id: Optional[int] = None
        self.room_type: Optional[str] = None
        self.last_train_date: Optional[pd.Timestamp] = None
        self.last_train_window: Optional[np.ndarray] = None  # (seq_len, feature_dim) scaled

    def _build_sequences(self, X_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Slide window: input (seq_len × feat_dim), target = price ở next step."""
        n = len(X_scaled)
        Xs, ys = [], []
        for i in range(n - self.seq_len):
            Xs.append(X_scaled[i:i + self.seq_len])
            ys.append(X_scaled[i + self.seq_len, 0])    # price is col 0
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    def fit(self, series: pd.Series, exog: pd.DataFrame,
            hotel_id: Optional[int] = None, room_type: Optional[str] = None,
            epochs: int = 200, lr: float = 1e-3, patience: int = 20,
            verbose: bool = False) -> "LSTMForecastModel":
        _require_torch()
        self.hotel_id = hotel_id
        self.room_type = room_type
        self.last_train_date = series.index[-1]

        # Combine price + exog into feature matrix
        feat_df = exog.copy()
        feat_df.insert(0, "price", series.values)
        X = feat_df.values.astype(np.float32)    # (n, feature_dim)
        self.feature_dim = X.shape[1]

        # Standardize per-feature
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean) / self.scaler_std

        # Build sequences
        Xs, ys = self._build_sequences(X_scaled)    # (n-seq, seq, feat), (n-seq,)
        # 80/20 train/val split (chronological, no shuffle)
        split = int(len(Xs) * 0.85)
        Xtr, ytr = Xs[:split], ys[:split]
        Xval, yval = Xs[split:], ys[split:]

        # Build model
        self.model = _LSTMNet(self.feature_dim, self.hidden_size,
                               self.num_layers, self.dropout)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        crit = nn.MSELoss()

        Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr)
        Xval_t = torch.from_numpy(Xval); yval_t = torch.from_numpy(yval)

        best_val = float("inf"); best_state = None; bad_epochs = 0
        for epoch in range(epochs):
            self.model.train()
            opt.zero_grad()
            pred = self.model(Xtr_t)
            loss = crit(pred, ytr_t)
            loss.backward()
            opt.step()

            # Val (no dropout for cleaner signal)
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xval_t)
                val_loss = crit(val_pred, yval_t).item()

            if val_loss < best_val - 1e-5:
                best_val = val_loss; best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    if verbose:
                        print(f"early stop at epoch {epoch}, best val {best_val:.4f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Store last seq_len days (scaled) làm starting window cho predict
        self.last_train_window = X_scaled[-self.seq_len:].copy()
        return self

    def predict(self, n_periods: int,
                exog_future: pd.DataFrame,
                alpha: float = 0.2,        # alpha=0.2 → 80% CI (p10/p90)
                index: Optional[pd.DatetimeIndex] = None,
                mc_samples: int = 50) -> ForecastResult:
        _require_torch()
        if self.model is None:
            raise RuntimeError("Model chưa fit. Gọi .fit() trước.")
        if index is None:
            start = self.last_train_date + pd.Timedelta(days=1)
            index = pd.date_range(start, periods=n_periods, freq="D")

        # Scale future exog
        # exog_future has only EXOG_COLS; price col will be filled iteratively
        # Build full feature row template (price + exog), scaled
        future_exog_scaled = (
            (exog_future.values - self.scaler_mean[1:]) / self.scaler_std[1:]
        ).astype(np.float32)    # (n_periods, n_exog)

        # MC Dropout: K parallel trajectories
        K = mc_samples
        device = next(self.model.parameters()).device

        # Init window: K copies of last_train_window
        window = torch.from_numpy(
            np.tile(self.last_train_window, (K, 1, 1))
        ).to(device).float()    # (K, seq_len, feat_dim)

        self.model.train()        # IMPORTANT: keep dropout active for MC sampling

        preds_scaled = np.zeros((K, n_periods), dtype=np.float32)
        with torch.no_grad():
            for t in range(n_periods):
                pred = self.model(window).cpu().numpy()    # (K,) — scaled price
                preds_scaled[:, t] = pred
                # Build next timestep feature: predicted_price (scaled) + exog (scaled)
                next_exog = future_exog_scaled[t]    # (n_exog,)
                next_step = np.concatenate([
                    pred.reshape(K, 1),               # K predicted prices
                    np.tile(next_exog, (K, 1)),       # K copies of same exog
                ], axis=1)                            # (K, feat_dim)
                next_step_t = torch.from_numpy(next_step).to(device).unsqueeze(1)  # (K, 1, feat)
                # Slide window
                window = torch.cat([window[:, 1:, :], next_step_t], dim=1)

        # Unscale predictions
        preds = preds_scaled * self.scaler_std[0] + self.scaler_mean[0]    # (K, n_periods)

        # Quantiles per timestep
        q_low, q_med, q_high = np.quantile(preds, [alpha / 2, 0.5, 1 - alpha / 2], axis=0)
        return ForecastResult(
            index=index,
            p10=pd.Series(q_low, index=index),
            p50=pd.Series(q_med, index=index),
            p90=pd.Series(q_high, index=index),
        )

    def save(self, path: str | Path) -> None:
        state = {
            "arch": {
                "seq_len": self.seq_len, "hidden_size": self.hidden_size,
                "num_layers": self.num_layers, "dropout": self.dropout,
                "feature_dim": self.feature_dim,
            },
            "state_dict": self.model.state_dict() if self.model is not None else None,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "hotel_id": self.hotel_id,
            "room_type": self.room_type,
            "last_train_date": self.last_train_date,
            "last_train_window": self.last_train_window,
        }
        joblib.dump(state, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecastModel":
        _require_torch()
        state = joblib.load(Path(path))
        arch = state["arch"]
        inst = cls(seq_len=arch["seq_len"], hidden_size=arch["hidden_size"],
                    num_layers=arch["num_layers"], dropout=arch["dropout"])
        inst.feature_dim = arch["feature_dim"]
        inst.scaler_mean = state["scaler_mean"]
        inst.scaler_std = state["scaler_std"]
        inst.hotel_id = state["hotel_id"]
        inst.room_type = state["room_type"]
        inst.last_train_date = state["last_train_date"]
        inst.last_train_window = state["last_train_window"]
        if state["state_dict"] is not None:
            inst.model = _LSTMNet(inst.feature_dim, inst.hidden_size,
                                   inst.num_layers, inst.dropout)
            inst.model.load_state_dict(state["state_dict"])
            inst.model.eval()
        return inst


def model_path_lstm(hotel_id: int, room_type_name: str,
                     models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"forecast_lstm_{hotel_id}_{_slugify(room_type_name)}.joblib"


def train_all_lstm(features_path: Path | str | None = None,
                    holdout_days: int = 30,
                    target_lead_time: int = 30,
                    tolerance: int = 5,
                    seed: int = 42,
                    models_dir: Path | str = MODELS_DIR) -> pd.DataFrame:
    """Fit 1 LSTM per (hotel, room_type), evaluate trên holdout 30 ngày."""
    from .forecast import build_series
    import time

    torch.manual_seed(seed); np.random.seed(seed)

    df = pd.read_parquet(features_path or DEFAULT_FEATURES)
    models_dir = Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for hotel_id in sorted(df["hotel_id"].unique()):
        sub = df[df["hotel_id"] == hotel_id]
        for room_type in sorted(sub["room_type_name"].unique()):
            target, exog = build_series(df, int(hotel_id), room_type,
                                         target_lead_time=target_lead_time,
                                         tolerance=tolerance)
            target = target.dropna()
            exog = exog.loc[target.index]
            train_y, hold_y = target.iloc[:-holdout_days], target.iloc[-holdout_days:]
            train_x, hold_x = exog.iloc[:-holdout_days], exog.iloc[-holdout_days:]

            t0 = time.time()
            m = LSTMForecastModel().fit(
                train_y, exog=train_x,
                hotel_id=int(hotel_id), room_type=room_type,
            )
            train_time = time.time() - t0

            fc = m.predict(holdout_days, exog_future=hold_x, index=hold_y.index)

            mape = float((np.abs(hold_y - fc.p50) / hold_y).mean())
            rmse = float(np.sqrt(((hold_y - fc.p50) ** 2).mean()))
            coverage = float(((hold_y >= fc.p10) & (hold_y <= fc.p90)).mean())

            path = model_path_lstm(int(hotel_id), room_type, models_dir)
            m.save(path)

            rows.append({
                "hotel_id": int(hotel_id),
                "room_type": room_type,
                "n_train": len(train_y),
                "n_hold": len(hold_y),
                "mape_pct": mape * 100,
                "rmse": rmse,
                "coverage80_pct": coverage * 100,
                "train_time_s": train_time,
                "path": str(path.relative_to(Path(__file__).resolve().parents[1])),
            })
    return pd.DataFrame(rows)
