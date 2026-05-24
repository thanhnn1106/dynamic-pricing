"""Dynamic Pricing — core package.

Modules:
    data      Load + clean hotel inventory CSV.
    features  Feature engineering (lead_time, calendar, occupancy, lag).
    forecast  SARIMAX price forecast per (hotel, room_type) — median + 80% CI.
    demand    Binary classifier — P(book | price, context).
    pricing   Optimize price = argmax_X (X * P(book | X)) on a grid.
"""

__version__ = "0.1.0"
