import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Columns where 0 is biologically impossible → treat as missing
ZERO_AS_NULL_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_AS_NULL_COLS:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
    df.drop_duplicates(inplace=True)
    return df


def get_summary(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "missing": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "class_balance": df['Outcome'].value_counts().to_dict() if 'Outcome' in df.columns else {},
        "describe": df.describe(),
    }


def split_and_scale(df: pd.DataFrame, target: str = 'Outcome', test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler, X.columns.tolist()
