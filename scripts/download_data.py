from __future__ import annotations

from pathlib import Path


INSTRUCTIONS = """
To obtain the dataset:
1) Download "Online Furniture Orders: Delivery & Assembly" from Kaggle.
2) Place the CSV as data/raw/furniture_orders_dataset.csv

Alternatively, configure Kaggle API and automate download.
"""


def main() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    print(INSTRUCTIONS)


if __name__ == "__main__":
    main()


