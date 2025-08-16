from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import load_pickle


def main(args: argparse.Namespace) -> None:
    model = load_pickle(args.model)
    df = pd.read_csv(args.input)

    # Use data_config to select the same feature columns used in training
    with open(args.data_config, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    feature_cols = data_cfg.get("categorical_columns", []) + data_cfg.get("numeric_columns", [])

    y_prob = model.predict_proba(df[feature_cols])[:, 1]
    out = df.copy()
    out["churn_probability"] = y_prob
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions with trained model")
    parser.add_argument("--model", default="models/churn_model.pkl")
    parser.add_argument("--input", default="data/processed/engineered_features.csv")
    parser.add_argument("--output", default="data/processed/model_predictions.csv")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    main(parser.parse_args())


