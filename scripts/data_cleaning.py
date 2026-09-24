import re
import pandas as pd


RAW_DATA_PATH = "cars.csv"
CLEANED_DATA_PATH = "cars_cleaned.csv"


# 1. Standardizarea denumirilor coloanelor

def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    new_columns = []

    for col in df.columns:
        clean_col = col.strip().lower()

        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")

        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")

        new_columns.append(clean_col)

    df.columns = new_columns

    return df


# 2. Eliminarea spatiilor inutile din valorile textuale

def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_columns = df.select_dtypes(include="str").columns

    for col in text_columns:
        df[col] = df[col].str.strip()

    return df


# 3. Standardizarea valorilor lipsa

MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
}


def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)

    return df


# 4. Standardizarea valorilor categoriale

def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    return df


# 5. Conversia coloanelor numerice

def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "priceusd",
        "year",
        "mileage_kilometers",
        "volume_cm3",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# 6. Tratarea kilometrajelor care nu par realiste

def _clean_invalid_mileage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    invalid_mileage = (
        (df["mileage_kilometers"] < 0)
        | (df["mileage_kilometers"] > 1_000_000)
    )

    df.loc[
        invalid_mileage,
        "mileage_kilometers"
    ] = pd.NA

    return df


# 7. Tratarea capacitatilor cilindrice care nu par realiste

def _clean_invalid_engine_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    invalid_engine_volume = (
        (df["volume_cm3"] < 500)
        | (df["volume_cm3"] > 7_000)
    )

    df.loc[
        invalid_engine_volume,
        "volume_cm3"
    ] = pd.NA

    return df


# 8. Eliminarea randurilor duplicate

def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    return df


# 9. Eliminarea randurilor fara valoarea tinta

def _remove_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=["priceusd"])

    return df


# 10. Pipeline-ul de curatare

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = (
        df
        .pipe(_standardize_column_names)
        .pipe(_strip_string_values)
        .pipe(_replace_missing_like_values)
        .pipe(_clean_categorical_values)
        .pipe(_convert_numeric_columns)
        .pipe(_clean_invalid_mileage)
        .pipe(_clean_invalid_engine_volume)
        .pipe(_remove_duplicates)
        .pipe(_remove_rows_with_missing_target)
        .reset_index(drop=True)
    )

    return df_clean


# 11. Rularea pipeline-ului

def main() -> None:
    print("Loading raw dataset...")

    df_raw = pd.read_csv(RAW_DATA_PATH)

    print(f"Rows before cleaning: {len(df_raw)}")

    print("Cleaning dataset...")

    df_cleaned = clean(df_raw)

    print(f"Rows after cleaning: {len(df_cleaned)}")

    print("Saving cleaned dataset...")

    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")


if __name__ == "__main__":
    main()






