from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


def _safe_add(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.fillna(0) + b.fillna(0)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: np.nan})
    return numerator / denom


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"shipping_cost", "assembly_cost"}.issubset(df.columns):
        df["total_delivery_cost"] = _safe_add(df["shipping_cost"], df["assembly_cost"])

    if {"total_amount", "delivery_window_days"}.issubset(df.columns):
        df["price_per_day"] = _safe_divide(df["total_amount"], df["delivery_window_days"])

    # Delivery complexity score (simple heuristic)
    if "assembly_service" in df.columns:
        assembly_component = df["assembly_service"].astype(int)
    else:
        assembly_component = 0
    if "delivery_window_days" in df.columns:
        window_component = pd.cut(df["delivery_window_days"], bins=[-np.inf, 1, 3, 7, np.inf], labels=[3, 2, 1, 0]).astype(int)
    else:
        window_component = 0
    if "product_category" in df.columns:
        # Map categories to complexity roughly by frequency-insensitive hash
        category_component = df["product_category"].astype(str).apply(lambda s: hash(s) % 3)
    else:
        category_component = 0
    try:
        df["delivery_complexity_score"] = assembly_component + window_component + category_component
    except Exception:
        pass

    # Customer value tier based on total amount
    if "total_amount" in df.columns:
        df["customer_value_tier"] = pd.cut(
            df["total_amount"], bins=[-np.inf, 100, 500, np.inf], labels=["Low", "Medium", "High"],
        )

    # Delivery efficiency: 1 if delivered, 0 otherwise
    if "delivery_status" in df.columns:
        delivered_like = df["delivery_status"].astype(str).str.lower().str.contains("deliver")
        df["delivery_efficiency"] = delivered_like.astype(int)

    return df


