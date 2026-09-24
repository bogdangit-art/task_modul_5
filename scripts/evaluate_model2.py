import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from data_preprocessing import split_features_and_target


DATA_PATH = "cars_cleaned_with_features.csv"
MODEL_PATH = "random_forest_model.joblib"


# 1. Incarcarea dataset-ului

df = pd.read_csv(DATA_PATH)


# 2. Separarea caracteristicilor si a coloanei tinta

X, y = split_features_and_target(df)


# 3. Impartirea datelor in train si test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Incarcarea modelului Random Forest

print("Loading Random Forest model...")

loaded_model = joblib.load(MODEL_PATH)


# 5. Realizarea predictiilor pe setul de test

print("Making predictions...")

y_pred = loaded_model.predict(X_test)

print(y_pred[:10])


# 6. Calcularea metricilor de regresie

print("Calculating regression metrics...")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})

print("\nRandom Forest regression metrics:")
print(metrics)


# 7. Crearea tabelului pentru analiza predictiilor

prediction_analysis = pd.DataFrame({
    "actual_price_usd": y_test.values,
    "predicted_price_usd": y_pred,
})


# 8. Calcularea erorilor

prediction_analysis["error_usd"] = (
    prediction_analysis["actual_price_usd"]
    - prediction_analysis["predicted_price_usd"]
)

prediction_analysis["absolute_error_usd"] = (
    prediction_analysis["error_usd"].abs()
)


# 9. Afisarea unor exemple

print("\nPrediction examples:")

print(
    prediction_analysis
    .sample(10, random_state=42)
)


# 10. Afisarea celor mai mari erori

print("\nLargest prediction errors:")

print(
    prediction_analysis
    .sort_values(
        "absolute_error_usd",
        ascending=False
    )
    .head(10)
)