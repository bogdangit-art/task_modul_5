import pandas as pd


CLEANED_DATA_PATH = "cars_cleaned.csv"
FEATURES_DATA_PATH = "cars_cleaned_with_features.csv"

REFERENCE_YEAR = 2019


# 1. Crearea caracteristicii pentru varsta masinii

def _add_car_age_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["car_age"] = REFERENCE_YEAR - df["year"]

    return df


# 2. Crearea kilometrajului mediu anual

def _add_mileage_per_year_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    age_for_calculation = df["car_age"].replace(0, 1)

    df["mileage_per_year"] = (
        df["mileage_kilometers"] / age_for_calculation
    )

    return df


# 3. Pipeline pentru feature engineering

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = (
        df
        .pipe(_add_car_age_feature)
        .pipe(_add_mileage_per_year_feature)
        .reset_index(drop=True)
    )

    return df_features


# 4. Rularea pipeline-ului

def main() -> None:
    print("Loading cleaned dataset...")

    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

    print("Building features...")

    df_features = build_features(df_cleaned)

    print("Saving feature-engineered dataset...")

    df_features.to_csv(FEATURES_DATA_PATH, index=False)

    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")


if __name__ == "__main__":
    main()


