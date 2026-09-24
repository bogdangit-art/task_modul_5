import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Definirea coloanei tinta si a coloanelor de intrare

TARGET_COLUMN = "priceusd"

NUMERIC_FEATURES = [
    "mileage_kilometers",
    "volume_cm3",
    "car_age",
    "mileage_per_year",
]

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
]


def get_all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


# 2. Separarea caracteristicilor de intrare si a coloanei tinta

def split_features_and_target(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:

    X = df[get_all_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


# 3. Preprocesarea coloanelor numerice

def _build_numeric_transformer() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    return numeric_transformer


# 4. Preprocesarea coloanelor categoriale

def _build_categorical_transformer() -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    return categorical_transformer


# 5. Combinarea transformarilor

def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                _build_numeric_transformer(),
                NUMERIC_FEATURES
            ),
            (
                "cat",
                _build_categorical_transformer(),
                CATEGORICAL_FEATURES
            ),
        ],
        remainder="drop"
    )

    return preprocessor