# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, fix data types, remove duplicates.
    Returns a cleaned DataFrame.
    """
    df = df.copy()

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Duplicates removed: {before - len(df)}")

    # Fill missing numeric values with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"[INFO] Filled {missing} missing in '{col}' with median")

    # Fill missing categorical values with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"[INFO] Filled {missing} missing in '{col}' with mode")

    print(f"[INFO] Clean data shape: {df.shape}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables for ML.
    Returns encoded DataFrame.
    """
    df = df.copy()

    # Binary encoding
    df["gender"]          = (df["gender"] == "Female").astype(int)
    df["internet_access"] = (df["internet_access"] == "Yes").astype(int)

    # Ordinal encoding — parent education has a natural order
    edu_order = {"None": 0, "School": 1, "Graduate": 2, "Postgraduate": 3}
    df["parent_education"] = df["parent_education"].map(edu_order)

    # Encode target: Pass=1, Fail=0
    df["performance_label"] = (df["performance_label"] == "Pass").astype(int)

    print("[INFO] Categorical features encoded successfully")
    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split into feature matrix X and target vector y.
    Drops non-feature columns.
    """
    drop_cols = ["student_id", "name", "final_grade", "performance_label"]
    X = df.drop(columns=drop_cols)
    y = df["performance_label"]
    print(f"[INFO] Features : {list(X.columns)}")
    print(f"[INFO] Target   : Pass={y.sum()} | Fail={len(y)-y.sum()}")
    return X, y


def scale_features(X_train, X_test):
    """
    Standardize features.
    IMPORTANT: fit only on train, transform both — prevents data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    print("[INFO] Features scaled with StandardScaler")
    return X_train_scaled, X_test_scaled, scaler