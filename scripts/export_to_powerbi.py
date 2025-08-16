from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    processed_dir = Path("data/processed")
    exports_dir = Path("data/exports")
    exports_dir.mkdir(parents=True, exist_ok=True)

    preds_path = processed_dir / "model_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError("Run the pipeline first to generate model_predictions.csv")

    df = pd.read_csv(preds_path)
    cols = [c for c in df.columns if c != "churn_probability"] + ["churn_probability"]
    df[cols].to_csv(exports_dir / "powerbi_data.csv", index=False)

    summary = df["churn_probability"].describe().to_frame("value").reset_index().rename(
        columns={"index": "metric"}
    )
    summary.to_csv(exports_dir / "summary_metrics.csv", index=False)


if __name__ == "__main__":
    main()


