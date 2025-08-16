from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessConfig:
    target_method: str = "method1"  # method1 | method2 | method3
    rating_threshold: float = 3.0


def load_raw_data(csv_path: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from {csv_path}")
    return pd.read_csv(csv_path)


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "brand" in df.columns:
        df["brand"] = df["brand"].fillna("Unknown")
    for col in ["shipping_cost", "assembly_cost"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    if "customer_rating" in df.columns:
        mean_rating = df["customer_rating"].mean()
        df["customer_rating"] = df["customer_rating"].fillna(mean_rating)
    return df


def define_churn(df: pd.DataFrame, config: PreprocessConfig) -> pd.Series:
    method = (config.target_method or "method1").lower()
    rating_thr = config.rating_threshold

    delivery_status = df["delivery_status"] if "delivery_status" in df.columns else pd.Series([np.nan] * len(df))
    rating = df["customer_rating"] if "customer_rating" in df.columns else pd.Series([np.nan] * len(df))

    if method == "method1":
        failed_like = delivery_status.fillna("").str.lower().isin(["failed", "rescheduled", "cancelled"])
        low_rating = rating.fillna(5).astype(float) < rating_thr
        churn = (failed_like & low_rating).astype(int)
    elif method == "method2":
        # Time/behavior based: customers with a single purchase are churn-risk
        if "customer_id" in df.columns:
            order_counts = df.groupby("customer_id")["customer_id"].transform("count")
            churn = (order_counts == 1).astype(int)
        else:
            churn = pd.Series([0] * len(df))
    elif method == "method3":
        churn = (rating.fillna(5).astype(float) < rating_thr).astype(int)
    else:
        raise ValueError(f"Unknown target_method: {method}")
    return churn


def preprocess_dataframe(raw_df: pd.DataFrame, config: Optional[PreprocessConfig] = None) -> pd.DataFrame:
    config = config or PreprocessConfig()
    df = impute_missing_values(raw_df)
    if "churn" not in df.columns:
        df["churn"] = define_churn(df, config)
    return df


