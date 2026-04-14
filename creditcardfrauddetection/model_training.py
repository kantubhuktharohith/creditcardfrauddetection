"""
model_training.py — Complete ML pipeline for Credit Card Fraud Detection
Trains 3 models (Logistic Regression, Random Forest, Decision Tree)
with SMOTE for class imbalance handling.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

from utils import load_and_preprocess, evaluate_model

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def train_all_models(data_path: str = None):
    """
    Full training pipeline:
    1. Load & preprocess data
    2. Train/test split (stratified)
    3. Apply SMOTE on training data
    4. Train Logistic Regression, Random Forest, Decision Tree
    5. Evaluate all models
    6. Save models & scalers to disk
    
    Returns: dict with models, metrics, scalers, test data, and dataframe info.
    """
    if data_path is None:
        data_path = DATA_PATH

    # ── Step 1: Load & Preprocess ──
    print("=" * 60)
    print("  CREDIT CARD FRAUD DETECTION — MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"\n📂 Loading dataset from: {data_path}")

    if not os.path.exists(data_path):
        print("\n❌ ERROR: Dataset not found!")
        print(f"   Expected path: {data_path}")
        print("   Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("   Place 'creditcard.csv' in the data/ folder.")
        sys.exit(1)

    df, X, y, amount_scaler, time_scaler = load_and_preprocess(data_path)

    print(f"   ✅ Loaded {len(df):,} transactions")
    print(f"   📊 Features: {X.shape[1]}")
    print(f"   🔴 Fraudulent: {y.sum():,} ({y.mean()*100:.3f}%)")
    print(f"   🟢 Normal: {(y == 0).sum():,} ({(1 - y.mean())*100:.3f}%)")

    # ── Step 2: Train/Test Split ──
    print(f"\n📐 Splitting data (Train: {(1-TEST_SIZE)*100:.0f}% | Test: {TEST_SIZE*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set:     {len(X_test):,} samples")

    # ── Step 3: Apply SMOTE ──
    print("\n⚖️  Applying SMOTE to balance training data...")
    print(f"   Before SMOTE — Normal: {(y_train == 0).sum():,} | Fraud: {(y_train == 1).sum():,}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"   After SMOTE  — Normal: {(y_train_resampled == 0).sum():,} | Fraud: {(y_train_resampled == 1).sum():,}")
    print("   ✅ Training data is now balanced!")

    # ── Step 4: Define Models ──
    models_config = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=20
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=20
        ),
    }

    # ── Step 5: Train & Evaluate ──
    trained_models = {}
    all_metrics = []

    for name, model in models_config.items():
        print(f"\n🔧 Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        trained_models[name] = model

        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

        print(f"   ✅ {name} trained successfully!")
        print(f"   📊 Accuracy: {metrics['Accuracy']}% | Precision: {metrics['Precision']}% | "
              f"Recall: {metrics['Recall']}% | F1: {metrics['F1-Score']}% | ROC-AUC: {metrics['ROC-AUC']}%")

    # ── Step 6: Save Models & Scalers ──
    print(f"\n💾 Saving models to {MODELS_DIR}/...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, filepath)
        print(f"   ✅ Saved {filename}")

    # Save scalers
    joblib.dump(amount_scaler, os.path.join(MODELS_DIR, "amount_scaler.pkl"))
    joblib.dump(time_scaler, os.path.join(MODELS_DIR, "time_scaler.pkl"))
    print("   ✅ Saved scalers")

    # Save feature names
    feature_names = list(X.columns)
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    print("   ✅ Saved feature names")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("=" * 60)
    metrics_df = pd.DataFrame(all_metrics)
    print(f"\n{metrics_df.to_string(index=False)}")

    best_model = max(all_metrics, key=lambda m: m["F1-Score"])
    print(f"\n🏆 Best Model (by F1-Score): {best_model['Model']} — F1: {best_model['F1-Score']}%")
    print("=" * 60)

    return {
        "models": trained_models,
        "metrics": all_metrics,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_before_smote": X_train,
        "y_train_before_smote": y_train,
        "X_train_resampled": X_train_resampled,
        "y_train_resampled": y_train_resampled,
        "amount_scaler": amount_scaler,
        "time_scaler": time_scaler,
        "feature_names": feature_names,
        "dataframe": df,
    }


# ─────────────────────────────────────────────
#  RUN STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_all_models()
