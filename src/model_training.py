from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .utils import ensure_dir, save_pickle, save_json, get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 42
    use_smote: bool = True
    cv_folds: int = 5


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_cols),
            ("numeric", numeric_transformer, numeric_cols),
        ]
    )
    return preprocessor


def get_models_and_params(random_state: int) -> Dict[str, Tuple[object, Dict[str, List]]]:
    return {
        "logreg": (
            LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None),
            {"clf__C": [0.1, 1.0, 10.0]}
        ),
        "rf": (
            RandomForestClassifier(random_state=random_state, class_weight="balanced"),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 20]}
        ),
        "gb": (
            GradientBoostingClassifier(random_state=random_state),
            {"clf__n_estimators": [200, 400], "clf__learning_rate": [0.05, 0.1]}
        ),
        "xgb": (
            XGBClassifier(
                random_state=random_state,
                objective="binary:logistic",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                eval_metric="logloss",
                n_jobs=0,
            ),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 4, 6]}
        ),
    }


def train_best_model(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: List[str],
    numeric_cols: List[str],
    config: TrainConfig,
    artifacts_dir: str = "models",
) -> Dict[str, object]:
    X = df[categorical_cols + numeric_cols]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    steps = [("preprocess", preprocessor)]
    if config.use_smote:
        steps.append(("smote", SMOTE(random_state=config.random_state)))

    best_score = -np.inf
    best_result: Dict[str, object] = {}

    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    scorer = make_scorer(f1_score)

    models_and_params = get_models_and_params(config.random_state)
    for name, (estimator, grid) in models_and_params.items():
        logger.info(f"Training model: {name}")
        pipeline = ImbPipeline(steps=steps + [("clf", estimator)])
        gs = GridSearchCV(pipeline, param_grid=grid, scoring=scorer, cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)

        score = gs.best_score_
        logger.info(f"Model {name} best CV F1: {score:.4f}")
        if score > best_score:
            best_score = score
            best_result = {
                "name": name,
                "model": gs.best_estimator_,
                "X_test": X_test,
                "y_test": y_test,
            }

    if not best_result:
        raise RuntimeError("No model was trained successfully.")

    ensure_dir(artifacts_dir)
    model_path = f"{artifacts_dir}/churn_model.pkl"
    save_pickle(best_result["model"], model_path)
    logger.info(f"Saved best model to {model_path}")

    # Save encoders and scaler separately from the pipeline for convenience
    preprocess = best_result["model"].named_steps.get("preprocess")
    if preprocess is not None:
        categorical_encoder = preprocess.named_transformers_["categorical"]
        numeric_scaler = preprocess.named_transformers_["numeric"]
        save_pickle(categorical_encoder, f"{artifacts_dir}/encoders.pkl")
        save_pickle(numeric_scaler, f"{artifacts_dir}/scaler.pkl")

    return best_result


