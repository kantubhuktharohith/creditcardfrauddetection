"""
utils.py — Shared preprocessing & visualization helpers
Credit Card Fraud Detection Mini Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import os

# ─────────────────────────────────────────────
#  COLOUR PALETTE  (premium dark-theme friendly)
# ─────────────────────────────────────────────
COLORS = {
    "primary": "#6C63FF",
    "secondary": "#FF6584",
    "success": "#00C9A7",
    "warning": "#FFD93D",
    "danger": "#FF4C4C",
    "bg_dark": "#0E1117",
    "text": "#FAFAFA",
    "grid": "#1E2130",
    "fraud": "#FF6584",
    "normal": "#6C63FF",
    "model_colors": ["#6C63FF", "#00C9A7", "#FF6584"],
}

MODEL_NAMES = ["Logistic Regression", "Random Forest", "Decision Tree"]


# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_preprocess(filepath: str):
    """
    Load the credit card dataset and preprocess it.
    - Scales 'Time' and 'Amount' using StandardScaler.
    - Returns processed DataFrame, feature matrix X, label vector y, and fitted scaler.
    """
    df = pd.read_csv(filepath)

    # Scale Time and Amount
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))

    # Create a dedicated scaler for prediction later (fits on Amount only)
    amount_scaler = StandardScaler()
    amount_scaler.fit(df["Amount"].values.reshape(-1, 1))

    time_scaler = StandardScaler()
    time_scaler.fit(df["Time"].values.reshape(-1, 1))

    # Drop original columns
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    # Reorder columns: scaled features first, then V1-V28, then Class
    cols = ["scaled_amount", "scaled_time"] + [f"V{i}" for i in range(1, 29)] + ["Class"]
    df = df[cols]

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return df, X, y, amount_scaler, time_scaler


# ─────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a model and return a metrics dictionary."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob) * 100, 2) if y_prob is not None else "N/A",
    }
    return metrics


# ─────────────────────────────────────────────
#  PLOTTING FUNCTIONS
# ─────────────────────────────────────────────
def set_plot_style():
    """Apply premium dark style to all matplotlib plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg_dark"],
            "axes.facecolor": COLORS["bg_dark"],
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )


def plot_class_distribution(y, title="Class Distribution"):
    """Bar chart — Fraud vs Normal transaction count."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = y.value_counts()
    labels = ["Normal", "Fraud"]
    colors = [COLORS["normal"], COLORS["fraud"]]

    bars = ax.bar(labels, [counts[0], counts[1]], color=colors, width=0.5, edgecolor="white", linewidth=0.5)

    # Add count labels on top of bars
    for bar, count in zip(bars, [counts[0], counts[1]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
            color=COLORS["text"],
        )

    ax.set_title(title, fontweight="bold", fontsize=16, pad=15)
    ax.set_ylabel("Number of Transactions", fontsize=12)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_confusion_matrices(models, X_test, y_test, model_names):
    """Side-by-side confusion matrix heatmaps for all models."""
    set_plot_style()
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, model, name, color in zip(axes, models, model_names, COLORS["model_colors"]):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
            annot_kws={"size": 14, "fontweight": "bold"},
            linewidths=2,
            linecolor=COLORS["bg_dark"],
        )
        ax.set_title(name, fontweight="bold", fontsize=14, color=color, pad=10)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)

    fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=18, y=1.02)
    plt.tight_layout()
    return fig


def plot_roc_curves(models, X_test, y_test, model_names):
    """Overlay ROC curves for all models on one chart."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, name, color in zip(models, model_names, COLORS["model_colors"]):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, lw=1.5, label="Random Guess")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontweight="bold", fontsize=16, pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.2, edgecolor=COLORS["grid"])
    ax.grid(alpha=0.1)
    plt.tight_layout()
    return fig


def plot_accuracy_comparison(metrics_list):
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1 across models."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = [m["Model"] for m in metrics_list]
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(model_names))
    width = 0.18
    colors = ["#6C63FF", "#00C9A7", "#FF6584", "#FFD93D"]

    for i, (key, color) in enumerate(zip(metric_keys, colors)):
        values = [m[key] for m in metrics_list]
        bars = ax.bar(x + i * width, values, width, label=key, color=color, edgecolor="white", linewidth=0.3)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=COLORS["text"],
            )

    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=16, pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, framealpha=0.2, edgecolor=COLORS["grid"])
    ax.grid(axis="y", alpha=0.1)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    """Horizontal bar chart of top N feature importances from a tree-based model."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    colors_gradient = plt.cm.cool(np.linspace(0.3, 0.9, top_n))

    ax.barh(
        range(top_n),
        importances[indices],
        color=colors_gradient,
        edgecolor="white",
        linewidth=0.3,
        height=0.7,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Top Feature Importances (Random Forest)", fontweight="bold", fontsize=16, pad=15)
    ax.grid(axis="x", alpha=0.1)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, top_n=15):
    """Correlation heatmap of top features correlated with the Class label."""
    set_plot_style()

    corr = df.corr()
    top_corr = corr["Class"].abs().sort_values(ascending=False).head(top_n + 1).index
    top_corr_matrix = df[top_corr].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        top_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5,
        linecolor=COLORS["bg_dark"],
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        f"Top {top_n} Features Correlated with Fraud",
        fontweight="bold",
        fontsize=16,
        pad=15,
    )
    plt.tight_layout()
    return fig
