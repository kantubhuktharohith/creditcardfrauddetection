"""
app.py — Premium Streamlit UI for Credit Card Fraud Detection
Features: Model Comparison, Real-Time Prediction, Data Insights, Visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
from model_training import train_all_models
from utils import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_accuracy_comparison,
    plot_feature_importance,
    plot_correlation_heatmap,
    COLORS,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Premium Dark Theme
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none !important;}

    /* Header Gradient with shimmer animation */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.35);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        animation: shimmer 4s ease-in-out infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes shimmer {
        0%, 100% { transform: translateX(-30%) translateY(-30%); }
        50% { transform: translateX(30%) translateY(30%); }
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(26, 26, 46, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 20px;
        padding: 1.8rem 1.2rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6C63FF, #00C9A7, #FF6584);
        border-radius: 20px 20px 0 0;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.02);
        border-color: rgba(108, 99, 255, 0.5);
        box-shadow: 0 16px 48px rgba(108, 99, 255, 0.2);
    }
    .metric-card:hover::after {
        opacity: 1;
    }
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #00C9A7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }

    /* Prediction Result Cards */
    .prediction-safe {
        background: linear-gradient(135deg, #00C9A7 0%, #00897B 100%);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(0, 201, 167, 0.35);
        animation: slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .prediction-fraud {
        background: linear-gradient(135deg, #FF6584 0%, #FF4C4C 100%);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(255, 101, 132, 0.35);
        animation: slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .prediction-safe::before, .prediction-fraud::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    }
    .prediction-safe h2, .prediction-fraud h2 {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        position: relative;
    }
    .prediction-safe p, .prediction-fraud p {
        color: rgba(255,255,255,0.95);
        font-size: 1rem;
        margin-top: 0.4rem;
        position: relative;
    }
    .confidence-bar {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.2);
        border-radius: 3px;
        margin-top: 0.8rem;
        overflow: hidden;
        position: relative;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        background: rgba(255,255,255,0.8);
        transition: width 1s ease;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(108, 99, 255, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 50%, #0E1117 100%);
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }

    /* Winner card */
    .model-winner {
        background: linear-gradient(135deg, rgba(0, 201, 167, 0.12) 0%, rgba(0, 201, 167, 0.04) 100%);
        border: 1px solid rgba(0, 201, 167, 0.3);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        animation: slideUp 0.5s ease;
    }

    /* Info Cards */
    .info-card {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .info-card:hover {
        border-color: rgba(108, 99, 255, 0.35);
        transform: translateY(-2px);
    }
    .info-card h4 {
        margin: 0 0 0.5rem 0;
        color: #6C63FF;
        font-size: 1rem;
    }
    .info-card p {
        margin: 0;
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Quick test sample cards */
    .sample-card {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    .sample-card.normal {
        border-color: rgba(0, 201, 167, 0.3);
    }
    .sample-card.fraud {
        border-color: rgba(255, 101, 132, 0.3);
    }
    .sample-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }

    /* Feature card for home page */
    .feature-item {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        padding: 0.7rem 0;
        border-bottom: 1px solid rgba(108, 99, 255, 0.08);
    }
    .feature-item:last-child {
        border-bottom: none;
    }
    .feature-icon {
        font-size: 1.5rem;
        min-width: 2rem;
        text-align: center;
    }
    .feature-text {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .feature-text strong {
        color: white;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 26, 46, 0.4);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #6C63FF, #764ba2) !important;
    }

    /* Form submit button */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6C63FF, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.4) !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Separator */
    .fancy-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(108, 99, 255, 0.3), transparent);
        margin: 2rem 0;
        border: none;
    }

    /* Animated number counter */
    .counter-row {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
    }

    /* How it works steps */
    .step-card {
        background: rgba(26, 26, 46, 0.5);
        border: 1px solid rgba(108, 99, 255, 0.1);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .step-card:hover {
        border-color: rgba(108, 99, 255, 0.3);
        transform: translateY(-3px);
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6C63FF, #764ba2);
        color: white;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .step-card h4 {
        margin: 0.3rem 0;
        color: white;
        font-size: 0.95rem;
    }
    .step-card p {
        margin: 0;
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  DATA & MODEL LOADING (cached)
# ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _models_exist_on_disk():
    """Check if all trained models are already saved."""
    required = ["logistic_regression.pkl", "random_forest.pkl", "decision_tree.pkl",
                "amount_scaler.pkl", "time_scaler.pkl", "feature_names.pkl"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def _load_from_disk(data_path):
    """Load pre-trained models from disk (fast: ~2 seconds)."""
    import pandas as pd
    from utils import load_and_preprocess
    from sklearn.model_selection import train_test_split
    from utils import evaluate_model

    # Load models
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    dt = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
    amount_scaler = joblib.load(os.path.join(MODELS_DIR, "amount_scaler.pkl"))
    time_scaler = joblib.load(os.path.join(MODELS_DIR, "time_scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    trained_models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Decision Tree": dt,
    }

    # Load & preprocess data for evaluation
    df, X, y, _, _ = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate
    all_metrics = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(metrics)

    return {
        "models": trained_models,
        "metrics": all_metrics,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_before_smote": X_train,
        "y_train_before_smote": y_train,
        "X_train_resampled": X_train,  # placeholder for display
        "y_train_resampled": y_train,  # placeholder
        "amount_scaler": amount_scaler,
        "time_scaler": time_scaler,
        "feature_names": feature_names,
        "dataframe": df,
    }


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load saved models if available, otherwise train from scratch."""
    data_path = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")
    if not os.path.exists(data_path):
        return None

    if _models_exist_on_disk():
        # FAST PATH: load pre-trained models (~2 seconds)
        return _load_from_disk(data_path)
    else:
        # SLOW PATH: train from scratch (first run only)
        results = train_all_models(data_path)
        return results


# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="text-align:center; padding: 1.2rem 0 0.8rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.3rem;">🛡️</div>
        <div style="font-size: 1.2rem; font-weight: 800; color: #6C63FF; letter-spacing: 0.5px;">FRAUD DETECTOR</div>
        <div style="font-size: 0.7rem; color: rgba(255,255,255,0.35); letter-spacing: 3px; margin-top: 0.2rem;">AI-POWERED</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📌 Navigate")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Model Comparison", "🔮 Predict Fraud", "📈 Data Insights"],
        label_visibility="collapsed",
    )

    
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📈 Quick Stats")
    st.markdown("""
- ✅ 3 Models Trained
- ✅ SMOTE Applied
- ✅ 284K Transactions
    """)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("""<p style="text-align:center; color:rgba(255,255,255,0.25); font-size:0.7rem; margin-top: 1rem;">
        Credit Card Fraud Detection<br>Mini Project © 2026
    </p>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
results = load_pipeline()

if results is None:
    st.markdown(
        """
    <div class="main-header">
        <h1>🛡️ Credit Card Fraud Detection</h1>
        <p>AI-powered fraud detection using multiple ML models</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.error("### ❌ Dataset Not Found!")
    st.warning(
        """
    **To get started:**
    1. Download the dataset from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    2. Place the `creditcard.csv` file in the `data/` folder
    3. Refresh this page

    The dataset contains **284,807 transactions** with **492 fraud cases**.
    """
    )
    st.stop()

# Unpack results
models = results["models"]
metrics_list = results["metrics"]
X_test = results["X_test"]
y_test = results["y_test"]
df = results["dataframe"]
feature_names = results["feature_names"]
amount_scaler = results["amount_scaler"]
time_scaler = results["time_scaler"]
model_list = list(models.values())
model_names = list(models.keys())


# ─────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown(
        """
    <div class="main-header">
        <h1>🛡️ Credit Card Fraud Detection</h1>
        <p>AI-powered fraud detection with multiple ML models, SMOTE balancing & real-time prediction</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Key Metrics
    total = len(df)
    fraud_count = int(df["Class"].sum())
    normal_count = total - fraud_count
    fraud_pct = fraud_count / total * 100
    best = max(metrics_list, key=lambda m: m["F1-Score"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">{total:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-icon">✅</div>
                <div class="metric-label">Normal</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #6C63FF, #667eea); -webkit-background-clip: text;">{normal_count:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-icon">🚨</div>
                <div class="metric-label">Fraudulent</div>
                <div class="metric-value" style="background: linear-gradient(135deg, #FF6584, #FF4C4C); -webkit-background-clip: text;">{fraud_count:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-label">Fraud Rate</div>
                <div class="metric-value">{fraud_pct:.3f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Two columns: chart + features
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-header">📊 Class Distribution</p>', unsafe_allow_html=True)

        # Interactive Plotly donut chart instead of static matplotlib
        fig = go.Figure(data=[go.Pie(
            labels=["Normal", "Fraud"],
            values=[normal_count, fraud_count],
            hole=0.6,
            marker=dict(colors=[COLORS["normal"], COLORS["fraud"]], line=dict(width=2, color="#0E1117")),
            textinfo="label+percent",
            textfont=dict(size=14, family="Inter"),
            hoverinfo="label+value+percent",
            pull=[0, 0.08],
        )])
        fig.update_layout(
            template="plotly_dark",
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            annotations=[dict(
                text=f"<b>{total:,}</b><br><span style='font-size:11px;color:rgba(255,255,255,0.5)'>Total</span>",
                x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white",
            )],
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">🎯 Project Highlights</p>', unsafe_allow_html=True)

        st.markdown("""
        - 🤖 **3 ML Models** — Logistic Regression, Random Forest, Decision Tree
        - ⚖️ **SMOTE Balancing** — Synthetic sampling to handle 0.17% fraud rate
        - 📊 **Rich Visualizations** — Confusion matrices, ROC curves, feature importance
        - 🔮 **Real-Time Prediction** — Input transaction details → instant fraud detection
        - 🏆 **Model Comparison** — Side-by-side metrics to find the best model
        """)

        st.markdown("")
        st.success(f"🏆 **Best Model:** {best['Model']} — **{best['F1-Score']}% F1-Score**")

    # How it works
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">⚙️ How It Works</p>', unsafe_allow_html=True)

    steps = [
        ("1", "Load Data", "284K transactions"),
        ("2", "Preprocess", "Scale features"),
        ("3", "SMOTE", "Balance classes"),
        ("4", "Train Models", "3 ML algorithms"),
        ("5", "Evaluate", "Compare metrics"),
        ("6", "Predict", "Real-time detection"),
    ]
    step_cols = st.columns(6)
    for i, (num, title, desc) in enumerate(steps):
        with step_cols[i]:
            st.markdown(
                f"""<div class="step-card">
                    <div class="step-number">{num}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )

    # Dataset Preview
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">📋 Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=300)


# ─────────────────────────────────────────────
#  PAGE: MODEL COMPARISON
# ─────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown(
        """
    <div class="main-header">
        <h1>📊 Model Comparison</h1>
        <p>Side-by-side evaluation of all trained models</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Model performance cards
    st.markdown('<p class="section-header">🏅 Model Scores</p>', unsafe_allow_html=True)
    best = max(metrics_list, key=lambda m: m["F1-Score"])

    model_cols = st.columns(3)
    model_icons = ["📐", "🌲", "🌳"]
    model_colors = ["#6C63FF", "#00C9A7", "#FF6584"]

    for i, m in enumerate(metrics_list):
        with model_cols[i]:
            is_best = m["Model"] == best["Model"]
            if is_best:
                st.markdown("🏆 **BEST MODEL**")
            st.markdown(f"### {model_icons[i]} {m['Model']}")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{m['Accuracy']}%")
            c2.metric("F1-Score", f"{m['F1-Score']}%")
            c3, c4 = st.columns(2)
            c3.metric("Precision", f"{m['Precision']}%")
            c4.metric("Recall", f"{m['Recall']}%")
            if m.get('ROC-AUC') and m['ROC-AUC'] != 'N/A':
                st.metric("ROC-AUC", f"{m['ROC-AUC']}%")

    # Winner callout
    st.markdown(
        f"""<div class="model-winner">
            <h3 style="margin:0;">🏆 Winner: {best['Model']}</h3>
            <p style="margin:0.3rem 0 0 0; color: rgba(255,255,255,0.7);">
                Best F1-Score of <strong>{best['F1-Score']}%</strong> — optimal balance of precision and recall for fraud detection.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # Charts in tabs
    tab1, tab2, tab3 = st.tabs(["📊 Accuracy Comparison", "🎯 Confusion Matrices", "📈 ROC Curves"])

    with tab1:
        # Interactive Plotly bar chart
        metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
        fig = go.Figure()
        colors = ["#6C63FF", "#00C9A7", "#FF6584", "#FFD93D"]
        for j, key in enumerate(metric_keys):
            fig.add_trace(go.Bar(
                name=key,
                x=[m["Model"] for m in metrics_list],
                y=[m[key] for m in metrics_list],
                text=[f"{m[key]}%" for m in metrics_list],
                textposition="auto",
                marker_color=colors[j],
                marker_line=dict(width=0),
            ))
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=13),
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 110]),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(t=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = plot_confusion_matrices(model_list, X_test, y_test, model_names)
        st.pyplot(fig, use_container_width=True)

    with tab3:
        fig = plot_roc_curves(model_list, X_test, y_test, model_names)
        st.pyplot(fig, use_container_width=True)

    # Detailed metrics table
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">📋 Detailed Metrics Table</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(metrics_list)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  PAGE: PREDICT FRAUD
# ─────────────────────────────────────────────
elif page == "🔮 Predict Fraud":
    st.markdown(
        """
    <div class="main-header">
        <h1>🔮 Real-Time Fraud Prediction</h1>
        <p>Enter transaction details to check if it's fraudulent</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Quick test presets
    st.markdown('<p class="section-header">⚡ Quick Test — Choose a Sample</p>', unsafe_allow_html=True)

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        use_normal = st.button("✅ Test Normal Transaction", use_container_width=True)
    with preset_col2:
        use_fraud = st.button("🚨 Test Suspicious Transaction", use_container_width=True)
    with preset_col3:
        use_custom = st.button("✏️ Custom Input", use_container_width=True)

    # Set defaults based on selection
    if use_fraud:
        defaults = {
            "amount": 9999.99, "time": 80000.0,
            "v": [-5.0, 3.5, -8.0, 6.0, -3.0, -2.5, -5.5, 1.0, -3.5, -8.0,
                  4.5, -12.0, -1.5, -15.0, -1.0, -6.0, -7.0, -2.5, 2.5, 0.5,
                  0.8, -0.5, -0.2, -0.5, 0.3, -0.8, 1.5, 0.3],
        }
    else:
        defaults = {
            "amount": 149.62, "time": 0.0,
            "v": [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.24, 0.09,
                  -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25,
                  -0.02, 0.28, -0.11, 0.01, 0.01, 0.00, -0.19, -0.15],
        }

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">💳 Transaction Details</p>', unsafe_allow_html=True)

    # Input form
    with st.form("prediction_form"):
        # Main inputs highlighted
        main_col1, main_col2 = st.columns(2)
        with main_col1:
            amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, max_value=30000.0, value=defaults["amount"], step=0.01)
        with main_col2:
            time_val = st.number_input("⏰ Time (seconds from first txn)", min_value=0.0, max_value=200000.0, value=defaults["time"], step=1.0)

        # V-features in expander for cleaner look
        with st.expander("🔧 Advanced Features (V1-V28) — PCA-transformed components", expanded=False):
            st.caption("These are Principal Component Analysis features from the original data. Default values represent a sample transaction.")
            col1, col2, col3, col4 = st.columns(4)
            v = defaults["v"]
            v_inputs = []
            for idx in range(28):
                target_col = [col1, col2, col3, col4][idx % 4]
                with target_col:
                    val = st.number_input(f"V{idx+1}", value=v[idx], format="%.4f", key=f"v{idx+1}")
                    v_inputs.append(val)

        st.markdown("")
        submitted = st.form_submit_button("🔍  Analyze Transaction", use_container_width=True)

    if submitted:
        # Prepare input
        scaled_amount = amount_scaler.transform([[amount]])[0][0]
        scaled_time = time_scaler.transform([[time_val]])[0][0]

        input_features = np.array([[scaled_amount, scaled_time] + v_inputs])

        st.markdown("")
        st.markdown('<p class="section-header">🎯 Prediction Results</p>', unsafe_allow_html=True)

        # Predict with all models
        cols = st.columns(len(models))

        for i, (name, model) in enumerate(models.items()):
            with cols[i]:
                prediction = model.predict(input_features)[0]
                confidence = model.predict_proba(input_features)[0] if hasattr(model, "predict_proba") else None

                if prediction == 0:
                    conf_pct = confidence[0] * 100 if confidence is not None else 0
                    st.markdown(
                        f"""<div class="prediction-safe">
                            <h2>✅ SAFE</h2>
                            <p><strong>{name}</strong></p>
                            <p>Confidence: {conf_pct:.1f}%</p>
                            <div class="confidence-bar"><div class="confidence-fill" style="width:{conf_pct}%"></div></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    conf_pct = confidence[1] * 100 if confidence is not None else 0
                    st.markdown(
                        f"""<div class="prediction-fraud">
                            <h2>🚨 FRAUD</h2>
                            <p><strong>{name}</strong></p>
                            <p>Confidence: {conf_pct:.1f}%</p>
                            <div class="confidence-bar"><div class="confidence-fill" style="width:{conf_pct}%"></div></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

        # Confidence comparison chart
        st.markdown("")
        fig = go.Figure()

        bar_colors = ["#6C63FF", "#00C9A7", "#FF6584"]
        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_features)[0]
                fig.add_trace(
                    go.Bar(
                        name=name,
                        x=["Normal", "Fraud"],
                        y=[prob[0] * 100, prob[1] * 100],
                        text=[f"{prob[0]*100:.1f}%", f"{prob[1]*100:.1f}%"],
                        textposition="auto",
                        marker_color=bar_colors[idx],
                        marker_line=dict(width=0),
                    )
                )

        fig.update_layout(
            title=dict(text="Prediction Confidence by Model", font=dict(size=16)),
            yaxis_title="Confidence (%)",
            barmode="group",
            template="plotly_dark",
            height=400,
            font=dict(family="Inter"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  PAGE: DATA INSIGHTS
# ─────────────────────────────────────────────
elif page == "📈 Data Insights":
    st.markdown(
        """
    <div class="main-header">
        <h1>📈 Data Insights</h1>
        <p>Deep dive into the dataset and model internals</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "⚖️ SMOTE Effect",
        "🔥 Feature Importance",
        "🔗 Correlations",
        "📊 Distribution Analysis",
    ])

    with tab1:
        st.markdown('<p class="section-header">⚖️ Class Balance — Before vs After SMOTE</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        y_train_before = results["y_train_before_smote"]
        y_train_after = results["y_train_resampled"]

        with col1:
            st.markdown("""<div class="info-card">
                <h4>❌ Before SMOTE</h4>
                <p>Training data is heavily imbalanced — model would be biased towards predicting "Normal"</p>
            </div>""", unsafe_allow_html=True)
            fig = plot_class_distribution(y_train_before, "Before SMOTE (Training Data)")
            st.pyplot(fig, use_container_width=True)
            st.metric("Normal", f"{(y_train_before == 0).sum():,}")
            st.metric("Fraud", f"{(y_train_before == 1).sum():,}")

        with col2:
            st.markdown("""<div class="info-card">
                <h4>✅ After SMOTE</h4>
                <p>Synthetic fraud samples generated — model can now learn fraud patterns effectively</p>
            </div>""", unsafe_allow_html=True)
            fig = plot_class_distribution(y_train_after, "After SMOTE (Training Data)")
            st.pyplot(fig, use_container_width=True)
            st.metric("Normal", f"{(y_train_after == 0).sum():,}")
            st.metric("Fraud", f"{(y_train_after == 1).sum():,}")

        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.info(
            "**SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic fraud samples "
            "by interpolating between existing fraud cases. This helps the model learn fraud patterns "
            "better without simply duplicating data. Note: SMOTE is applied **only on training data** "
            "to prevent data leakage."
        )

    with tab2:
        st.markdown('<p class="section-header">🔥 Feature Importance (Random Forest)</p>', unsafe_allow_html=True)
        st.markdown("""<div class="info-card">
            <h4>What are feature importances?</h4>
            <p>Feature importance shows how much each input contributes to the model's fraud detection decisions. 
            Higher values = more influential. The top features are the strongest signals for identifying fraud.</p>
        </div>""", unsafe_allow_html=True)

        rf_model = models.get("Random Forest")
        if rf_model is not None:
            # Interactive Plotly version
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-15:]
            top_features = [feature_names[i] for i in indices]
            top_values = importances[indices]

            fig = go.Figure(go.Bar(
                x=top_values,
                y=top_features,
                orientation='h',
                marker=dict(
                    color=top_values,
                    colorscale=[[0, '#6C63FF'], [0.5, '#00C9A7'], [1, '#FF6584']],
                    line=dict(width=0),
                ),
                text=[f"{v:.4f}" for v in top_values],
                textposition="auto",
            ))
            fig.update_layout(
                title="Top 15 Feature Importances",
                template="plotly_dark",
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                xaxis_title="Importance",
                margin=dict(l=100),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown(
            '<p class="section-header">🔗 Top Feature Correlations with Fraud</p>',
            unsafe_allow_html=True,
        )
        st.markdown("""<div class="info-card">
            <h4>Reading the heatmap</h4>
            <p>Red = positive correlation (increases with fraud), Blue = negative correlation (decreases with fraud). 
            Features with strong absolute correlation are the most useful for detecting fraud.</p>
        </div>""", unsafe_allow_html=True)
        fig = plot_correlation_heatmap(df, top_n=12)
        st.pyplot(fig, use_container_width=True)

    with tab4:
        st.markdown(
            '<p class="section-header">📊 Transaction Amount Distribution</p>',
            unsafe_allow_html=True,
        )

        # Amount distribution by class - overlaid histogram
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df[df["Class"] == 0]["scaled_amount"],
                name="Normal",
                marker_color=COLORS["normal"],
                opacity=0.7,
                nbinsx=50,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=df[df["Class"] == 1]["scaled_amount"],
                name="Fraud",
                marker_color=COLORS["fraud"],
                opacity=0.7,
                nbinsx=50,
            )
        )
        fig.update_layout(
            title="Scaled Amount Distribution (Normal vs Fraud)",
            barmode="overlay",
            template="plotly_dark",
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            xaxis_title="Scaled Amount",
            yaxis_title="Count",
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Basic statistics
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">📋 Dataset Statistics</p>', unsafe_allow_html=True)
        st.dataframe(df.describe().T, use_container_width=True)
