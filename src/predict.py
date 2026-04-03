# src/predict.py
# Prediction system — takes user input, returns prediction

import pandas as pd
import numpy as np
from src.model import load_model


# ── Column order must match training exactly ──────────────
FEATURE_NAMES = [
    "age", "gender", "attendance_pct", "study_hours_per_day",
    "prev_grade", "assignments_submitted",
    "parent_education", "internet_access"
]

# Encoding maps — must match preprocessing.py exactly
GENDER_MAP    = {"Male": 0, "Female": 1}
INTERNET_MAP  = {"No": 0, "Yes": 1}
EDU_MAP       = {"None": 0, "School": 1, "Graduate": 2, "Postgraduate": 3}


def encode_input(raw_input: dict) -> pd.DataFrame:
    """
    Convert raw user input (strings/numbers) into
    encoded DataFrame ready for the model.
    """
    encoded = {
        "age"                   : int(raw_input["age"]),
        "gender"                : GENDER_MAP[raw_input["gender"]],
        "attendance_pct"        : float(raw_input["attendance_pct"]),
        "study_hours_per_day"   : float(raw_input["study_hours_per_day"]),
        "prev_grade"            : float(raw_input["prev_grade"]),
        "assignments_submitted" : int(raw_input["assignments_submitted"]),
        "parent_education"      : EDU_MAP[raw_input["parent_education"]],
        "internet_access"       : INTERNET_MAP[raw_input["internet_access"]]
    }
    # Return as single-row DataFrame with correct column order
    return pd.DataFrame([encoded], columns=FEATURE_NAMES)


def predict_student(raw_input: dict, model_path: str = "models/model.pkl") -> dict:
    """
    Main prediction function.
    Takes raw input dict, returns prediction results dict.
    """
    # Load model, scaler, feature names
    model, scaler, feature_names = load_model(model_path)

    # Encode input
    input_df = encode_input(raw_input)

    # Scale using the SAME scaler used during training
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction    = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    label         = "Pass" if prediction == 1 else "Fail"
    pass_prob     = round(probabilities[1] * 100, 2)
    fail_prob     = round(probabilities[0] * 100, 2)
    confidence    = max(pass_prob, fail_prob)

    # Risk level based on pass probability
    if pass_prob >= 75:
        risk_level = "Low Risk"
    elif pass_prob >= 50:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return {
        "prediction"  : label,
        "pass_prob"   : pass_prob,
        "fail_prob"   : fail_prob,
        "confidence"  : confidence,
        "risk_level"  : risk_level
    }


def predict_batch(df_raw: pd.DataFrame,
                  model_path: str = "models/model.pkl") -> pd.DataFrame:
    """
    Predict for multiple students at once.
    Used in Streamlit dashboard to flag at-risk students.
    """
    model, scaler, feature_names = load_model(model_path)
    results = []

    for _, row in df_raw.iterrows():
        raw_input = row.to_dict()
        input_df  = encode_input(raw_input)
        scaled    = scaler.transform(input_df)
        pred      = model.predict(scaled)[0]
        prob      = model.predict_proba(scaled)[0]

        results.append({
            "prediction" : "Pass" if pred == 1 else "Fail",
            "pass_prob"  : round(prob[1] * 100, 2),
            "fail_prob"  : round(prob[0] * 100, 2),
            "risk_level" : (
                "Low Risk"    if prob[1] >= 0.75 else
                "Medium Risk" if prob[1] >= 0.50 else
                "High Risk"
            )
        })

    return pd.DataFrame(results)


def print_prediction_report(raw_input: dict, result: dict):
    """Print a clean, readable prediction report to terminal."""
    print("\n" + "="*50)
    print(" STUDENT PERFORMANCE PREDICTION")
    print("="*50)
    print(f"  Age                 : {raw_input['age']}")
    print(f"  Gender              : {raw_input['gender']}")
    print(f"  Attendance          : {raw_input['attendance_pct']}%")
    print(f"  Study Hours/Day     : {raw_input['study_hours_per_day']}")
    print(f"  Previous Grade      : {raw_input['prev_grade']}")
    print(f"  Assignments Done    : {raw_input['assignments_submitted']}/10")
    print(f"  Parent Education    : {raw_input['parent_education']}")
    print(f"  Internet Access     : {raw_input['internet_access']}")
    print("-"*50)
    print(f"  Prediction          : {result['prediction']}")
    print(f"  Pass Probability    : {result['pass_prob']}%")
    print(f"  Fail Probability    : {result['fail_prob']}%")
    print(f"  Confidence          : {result['confidence']}%")
    print(f"  Risk Level          : {result['risk_level']}")
    print("="*50)