# train.py — Run this to train and save the model

import pandas as pd
from src.preprocessing import (
    load_data, clean_data, encode_features,
    get_features_and_target, scale_features
)
from src.model import split_data, train_model, evaluate_model, save_model

print("="*55)
print(" STUDENT PERFORMANCE — MODEL TRAINING")
print("="*55)

# Step 1: Load and preprocess
df       = load_data("data/students.csv")
df       = clean_data(df)
df       = encode_features(df)
X, y     = get_features_and_target(df)

# Step 2: Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Step 3: Scale
X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

# Step 4: Train
model = train_model(X_train_sc, y_train)

# Step 5: Evaluate
feature_names = list(X.columns)
metrics = evaluate_model(model, X_test_sc, y_test, feature_names)

# Step 6: Save
save_model(model, scaler, feature_names)

print("\n" + "="*55)
print(f" TRAINING COMPLETE")
print(f" Accuracy : {metrics['accuracy']*100:.2f}%")
print(f" ROC-AUC  : {metrics['roc_auc']:.4f}")
print("="*55)
