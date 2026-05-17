"""Load + clean the SAVVY inventory CSV.

Raw schema (13 cols):
    updated_date, date, hotel_id, hotel_name, room_type_name,
    total_booked, total_maintenance, total, available,
    price, ota_price, room_type_segment, brand_sub_segment

Sẽ được điền ở Bước 1 (EDA) sau khi xác nhận:
- dtypes phù hợp (parse_dates cho updated_date + date)
- xử lý missing / duplicate
- type validation cho price/available
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "SAVVY-2BT.csv"


def load_raw(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Load CSV với dtypes phù hợp. TODO: hoàn thiện ở Bước 1."""
    df = pd.read_csv(
        path,
        parse_dates=["updated_date", "date"],
        low_memory=False,
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop dup + drop post-stay snapshots."""
    df = df.drop_duplicates()
    # updated_date > date = snapshot chụp sau đêm khách ở. Khách không thể book
    # đêm đã qua → mọi thay đổi total_booked là admin (no-show, cancel, refund),
    # không phải demand response to price. Giữ lại sẽ làm méo label did_book.
    df = df[df['date'] >= df['updated_date']].reset_index(drop=True)
    return df


def train_val_split(df: pd.DataFrame, val_days: int = 30):
    """Time-based split — KHÔNG random.

    Cuối series → validation, đầu series → training.
    TODO: hoàn thiện sau khi feature engineering xong (Bước 2-3).
    """
    raise NotImplementedError("Sẽ điền ở Bước 3 sau khi quyết định cutoff.")
