"""Load + clean hotel inventory CSV (sanitized cho public POC).

Raw schema (13 cols):
    updated_date, date, hotel_id, hotel_name, room_type_name,
    total_booked, total_maintenance, total, available,
    price, ota_price, room_type_segment, brand_sub_segment

CSVs (đã sanitize hotel_name):
    SAMV-HBT.csv  — hotel_id=956, 55K rows, 4 tháng updated_date
    SIMV-HBT.csv  — hotel_id=28,  376K rows, 18 tháng updated_date

`load_raw()` combine cả 2 mặc định. File gốc workspace không commit.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_CSVS = [
    DATA_DIR / "SAMV-HBT.csv",
    DATA_DIR / "SIMV-HBT.csv",
]
# Backwards-compat alias — vài chỗ vẫn import DEFAULT_CSV.
DEFAULT_CSV = DEFAULT_CSVS[0]


def load_raw(paths: str | Path | list = None) -> pd.DataFrame:
    """Load 1 hoặc nhiều CSV → concat thành 1 DataFrame.

    Args:
        paths: None (default — load tất cả DEFAULT_CSVS có tồn tại),
               str/Path (1 file), hoặc list của paths.
    """
    if paths is None:
        paths = DEFAULT_CSVS
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy CSV nào — kiểm tra {DATA_DIR}")

    dfs = [
        pd.read_csv(p, parse_dates=["updated_date", "date"], low_memory=False)
        for p in paths
    ]
    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop dup + post-stay + price=0 + room name typo. Fill NaN segment."""
    df = df.drop_duplicates()
    # updated_date > date = snapshot chụp sau đêm khách ở. Khách không thể book
    # đêm đã qua → mọi thay đổi total_booked là admin (no-show, cancel, refund),
    # không phải demand response to price. Giữ lại sẽ làm méo label did_book.
    df = df[df['date'] >= df['updated_date']]
    # SIMV "Standard" room có price=0 100% — không bookable, drop.
    df = df[df['price'] > 0]
    # SIMV có typo case: "Studio With Balcony" (1208 rows) vs "Studio with
    # balcony" (40575 rows). Drop bản hiếm để khỏi tạo series quá ngắn.
    df = df[df['room_type_name'] != 'Studio With Balcony']
    df = df.reset_index(drop=True)
    # SIMV ~7% NaN ở segment columns. Fill 'Unknown' để demand 1-hot OK.
    df['room_type_segment'] = df['room_type_segment'].fillna('Unknown')
    df['brand_sub_segment'] = df['brand_sub_segment'].fillna('Unknown')
    return df


def train_val_split(df: pd.DataFrame, val_days: int = 30):
    raise NotImplementedError("Sẽ điền ở Bước 3 sau khi quyết định cutoff.")
