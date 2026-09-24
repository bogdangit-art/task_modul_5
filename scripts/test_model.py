import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from data_preprocessing import split_features_and_target


DATA_PATH = "cars_cleaned_with_features.csv"
MODEL_PATH = "linear_regression_model.joblib"


# 1. Incarcarea dataset-ului

df = pd.read_csv(DATA_PATH)


# 2. Separarea caracteristicilor si a coloanei tinta

X, y = split_features_and_target(df)


# 3. Recrearea aceleiasi impartiri train/test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Incarcarea modelului antrenat

print("Loading model...")

loaded_model = joblib.load(MODEL_PATH)


# 5. Selectarea unor exemple din setul de test

print("Selecting samples...")

sample_X = X_test.sample(10, random_state=42)
sample_y = y_test.loc[sample_X.index]


# 6. Realizarea predictiilor

print("Making predictions...")

sample_predictions = loaded_model.predict(sample_X)


# 7. Compararea preturilor reale cu cele prezise

prediction_preview = pd.DataFrame({
    "actual_price_usd": sample_y.values,
    "predicted_price_usd": sample_predictions,
})

print("\nPrediction examples:")
print(prediction_preview)