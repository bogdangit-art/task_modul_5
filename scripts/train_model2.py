import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    split_features_and_target,
    build_preprocessor,
)


DATA_PATH = "cars_cleaned_with_features.csv"
MODEL_PATH = "random_forest_model.joblib"


# 1. Incarcarea setului de date pregatit

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


# 2. Separarea caracteristicilor de intrare si a coloanei tinta

print("Splitting features and target...")

X, y = split_features_and_target(df)

print("X shape:", X.shape)
print("y shape:", y.shape)


# 3. Impartirea datelor in set de antrenare si set de test

print("Splitting data into training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])


# 4. Crearea pipeline-ului complet de machine learning

print("Creating model pipeline...")

model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "regressor",
            RandomForestRegressor(
                random_state=42
            )
        ),
    ]
)


# 5. Antrenarea modelului Random Forest

print("Training Random Forest model...")

model.fit(X_train, y_train)


# 6. Salvarea modelului antrenat

print("Saving model...")

joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")