# src/model.py
# Model training, evaluation, and saving

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.metrics          import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# ── Train / Test Split ────────────────────────────────────
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split into train and test sets.
    80% train, 20% test — standard practice.
    stratify=y ensures both splits have same Pass/Fail ratio.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # ← critical for imbalanced classes
    )
    print(f"[INFO] Train size : {X_train.shape[0]} samples")
    print(f"[INFO] Test size  : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ── Train Model ───────────────────────────────────────────
def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.
    Random Forest is robust, handles non-linear relationships,
    and gives feature importances — perfect for this project.
    """
    model = RandomForestClassifier(
        n_estimators=100,       # 100 decision trees
        max_depth=6,            # prevent overfitting
        min_samples_split=5,    # minimum samples to split a node
        random_state=42,        # reproducibility
        class_weight="balanced" # handles Pass/Fail imbalance
    )
    model.fit(X_train, y_train)
    print("[INFO] Random Forest model trained successfully")
    return model


# ── Evaluate Model ────────────────────────────────────────
def evaluate_model(model, X_test, y_test, feature_names):
    """
    Full evaluation:
    - Accuracy
    - Classification report (precision, recall, F1)
    - Confusion matrix
    - ROC-AUC score
    - Cross-validation score
    - Feature importance chart
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*55)
    print(" MODEL EVALUATION REPORT")
    print("="*55)

    # 1. Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy  : {acc:.4f} ({acc*100:.2f}%)")

    # 2. ROC-AUC
    auc = roc_auc_score(y_test, y_prob)
    print(f"✅ ROC-AUC   : {auc:.4f}")

    # 3. Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    # 4. Cross-validation (5-fold)
    print("--- Cross-Validation (5-fold) ---")
    cv_scores = cross_val_score(model, 
                                pd.concat([pd.DataFrame(X_test), 
                                           pd.DataFrame(X_test)]),
                                np.concatenate([y_test, y_test]),
                                cv=5, scoring="accuracy")
    print(f"CV Scores : {cv_scores.round(3)}")
    print(f"CV Mean   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 5. Confusion Matrix chart
    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fail", "Pass"],
                yticklabels=["Fail", "Pass"],
                linewidths=1, ax=axes[0])
    axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Actual", fontsize=11)

    # 6. Feature Importance chart
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_features    = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    colors = sns.color_palette("viridis", len(sorted_features))
    axes[1].barh(sorted_features[::-1], sorted_importances[::-1],
                 color=colors[::-1], edgecolor="white")
    axes[1].set_title("Feature Importances", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Importance Score", fontsize=11)

    plt.tight_layout()
    plt.savefig("data/chart7_model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n[SAVED] data/chart7_model_evaluation.png")

    return {
        "accuracy"  : round(acc, 4),
        "roc_auc"   : round(auc, 4),
        "cv_mean"   : round(cv_scores.mean(), 4),
        "cv_std"    : round(cv_scores.std(), 4),
        "y_pred"    : y_pred,
        "y_prob"    : y_prob
    }


# ── Save Model ────────────────────────────────────────────
def save_model(model, scaler, feature_names, filepath="models/model.pkl"):
    """
    Save model + scaler + feature names together.
    Always save scaler with model — they must be used together.
    """
    os.makedirs("models", exist_ok=True)
    payload = {
        "model"         : model,
        "scaler"        : scaler,
        "feature_names" : feature_names
    }
    joblib.dump(payload, filepath)
    size = os.path.getsize(filepath) / 1024
    print(f"[INFO] Model saved → {filepath} ({size:.1f} KB)")


# ── Load Model ────────────────────────────────────────────
def load_model(filepath="models/model.pkl"):
    """Load model payload from disk."""
    payload = joblib.load(filepath)
    print(f"[INFO] Model loaded ← {filepath}")
    return payload["model"], payload["scaler"], payload["feature_names"]