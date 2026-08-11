from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset_v1_1.csv"
)

TEST_YEAR = 2025
TARGET = "transfer_fee"
DATE_COLUMN = "transfer_date"

RANDOM_STATE = 42


NUMERIC_FEATURES = [
    "age_at_transfer",
    "height",
    "matches",
    "started",
    "goals",
    "assists",
    "minutes",
    "rating",
    "is_same_league",
    "is_top5_destination",
    "goals_per90",
    "assists_per90",
    "goal_contributions_per90",
    "starts_ratio",
    "minutes_per_match",
    "age_squared",
]

CATEGORICAL_FEATURES = [
    "from_league_id",
    "to_league_id",
    "main_position",
    "foot",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[DATE_COLUMN, TARGET]
    ).copy()

    df = df[
        df[TARGET] > 0
    ].copy()

    df["transfer_year"] = (
        df[DATE_COLUMN].dt.year
    )

    return df


def create_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


df = load_data()

train_df = df[
    df["transfer_year"] <= 2023
].copy()

valid_df = df[
    df["transfer_year"] == 2024
].copy()

test_df = df[
    df["transfer_year"] == 2025
].copy()

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_valid = valid_df[FEATURES]
y_valid = valid_df[TARGET]


def objective(trial):
    params = {
        "objective": "reg:squarederror",

        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            1200,
            step=100,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.08,
            log=True,
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            5,
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            8,
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.65,
            1.0,
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.65,
            1.0,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            0.0,
            1.0,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.5,
            3.0,
        ),

        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    model = XGBRegressor(
        **params
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    y_train_log = np.log1p(
        y_train
    )

    pipeline.fit(
    X_train,
    np.log1p(y_train),
)

    prediction_log = pipeline.predict(
        X_valid
    )

    predictions = np.expm1(
        prediction_log
    )

    mae = mean_absolute_error(
        y_valid,
        predictions,
    )
    r2 = r2_score(
    y_valid,
    predictions,
)

    trial.set_user_attr(
        "r2",
        float(r2),
    )

    return mae


def main():
    print("=" * 60)
    print("Optuna XGBoost Tuning V1.1")
    print("=" * 60)

    print(
        f"Train: {len(train_df):,}"
    )

    print(
        f"Test : {len(test_df):,}"
    )

    study = optuna.create_study(
        direction="minimize",
    )

    study.optimize(
        objective,
        n_trials=50,
    )

    print()
    print("=" * 60)
    print("Best Result")
    print("=" * 60)

    print(
        f"Best MAE: "
        f"€{study.best_value:,.0f}"
    )

    print(
        f"Best R²: "
        f"{study.best_trial.user_attrs['r2']:.4f}"
    )

    print("\nBest Parameters:")

    for key, value in (
        study.best_params.items()
    ):
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()