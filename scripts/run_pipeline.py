from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Ensure project root is on sys.path when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_preprocessing import PreprocessConfig, preprocess_dataframe, load_raw_data
from src.feature_engineering import create_features
from src.model_training import TrainConfig, train_best_model
from src.model_evaluation import evaluate_predictions
from src.utils import ensure_dir, save_json, get_logger


logger = get_logger("run_pipeline")


def load_configs(data_config_path: str, model_config_path: str):
    with open(data_config_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg


def main(args: argparse.Namespace) -> None:
    data_cfg, model_cfg = load_configs(args.data_config, args.model_config)

    raw_csv = data_cfg["raw_csv"]
    processed_dir = Path("data/processed")
    exports_dir = Path("data/exports")
    ensure_dir(processed_dir)
    ensure_dir(exports_dir)

    df_raw = load_raw_data(raw_csv)
    df_clean = preprocess_dataframe(df_raw, PreprocessConfig(
        target_method=model_cfg.get("target_method", "method1"),
        rating_threshold=model_cfg.get("rating_threshold", 3.0),
    ))
    df_feat = create_features(df_clean)

    cleaned_path = processed_dir / "cleaned_data.csv"
    feat_path = processed_dir / "engineered_features.csv"
    df_clean.to_csv(cleaned_path, index=False)
    df_feat.to_csv(feat_path, index=False)
    logger.info(f"Saved processed data to {cleaned_path} and {feat_path}")

    categorical_cols = data_cfg.get("categorical_columns", [])
    numeric_cols = data_cfg.get("numeric_columns", [])
    target_col = data_cfg.get("target_column", "churn")

    train_result = train_best_model(
        df=df_feat,
        target_col=target_col,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        config=TrainConfig(
            test_size=model_cfg.get("test_size", 0.2),
            random_state=model_cfg.get("random_state", 42),
            use_smote=model_cfg.get("use_smote", True),
            cv_folds=model_cfg.get("cv_folds", 5),
        ),
        artifacts_dir="models",
    )

    best_model = train_result["model"]
    X_test = train_result["X_test"]
    y_test = train_result["y_test"]
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, y_prob)
    metadata = {
        "best_model": train_result["name"],
        "metrics": metrics,
    }
    save_json(metadata, "models/model_metadata.json")
    logger.info("Saved model metadata.")

    # Predictions for whole dataset for dashboard
    y_all_prob = best_model.predict_proba(df_feat[categorical_cols + numeric_cols])[:, 1]
    df_preds = df_feat.copy()
    df_preds["churn_probability"] = y_all_prob
    df_preds.to_csv(processed_dir / "model_predictions.csv", index=False)

    # Basic exports for Power BI
    df_preds[["churn_probability", target_col] + categorical_cols + numeric_cols].to_csv(
        exports_dir / "powerbi_data.csv", index=False
    )
    summary = df_preds["churn_probability"].describe().to_frame("value").reset_index().rename(
        columns={"index": "metric"}
    )
    summary.to_csv(exports_dir / "summary_metrics.csv", index=False)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn prediction pipeline")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml")
    main(parser.parse_args())


