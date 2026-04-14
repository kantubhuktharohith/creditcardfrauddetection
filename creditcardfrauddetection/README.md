# 🛡️ Credit Card Fraud Detection

> AI-powered fraud detection system using multiple Machine Learning models with SMOTE balancing and a premium Streamlit web interface.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **🤖 3 ML Models** | Logistic Regression, Random Forest, Decision Tree — trained & compared |
| **⚖️ SMOTE** | Synthetic Minority Over-sampling to handle extreme class imbalance (0.17% fraud) |
| **📊 Rich Visualizations** | Confusion matrices, ROC curves, accuracy comparison, feature importance, correlation heatmaps |
| **🔮 Real-Time Prediction** | Enter transaction details → instant fraud detection from all 3 models |
| **🌐 Streamlit Web UI** | Premium dark-themed interface with 4 pages |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **ML Framework:** Scikit-Learn
- **Imbalance Handling:** imbalanced-learn (SMOTE)
- **Web UI:** Streamlit
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Data Processing:** Pandas, NumPy

---

## 📁 Project Structure

```
creditcardfrauddetection/
├── app.py                  # Streamlit UI (main entry point)
├── model_training.py       # Model training pipeline
├── utils.py                # Preprocessing & visualization helpers
├── requirements.txt        # Dependencies
├── README.md               # This file
├── models/                 # Saved trained models (.pkl)
└── data/                   # Place creditcard.csv here
```

---

## 🚀 Setup & Run

### 1. Download the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will automatically:
- Load & preprocess the dataset
- Apply SMOTE to balance training data
- Train all 3 models
- Launch the web UI at `http://localhost:8501`

---

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~97%     | ~6%       | ~92%   | ~11%     | ~97%    |
| Random Forest       | ~99%     | ~92%      | ~82%   | ~87%     | ~97%    |
| Decision Tree       | ~99%     | ~73%      | ~78%   | ~75%     | ~88%    |

> *Exact values depend on the random state and SMOTE sampling.*  
> **🏆 Best Model:** Random Forest (highest F1-Score)

---

## 📸 Screenshots

### Home Page
- Dataset overview with key metrics
- Fraud vs Normal distribution chart

### Model Comparison
- Side-by-side metrics table
- Confusion matrices, ROC curves, accuracy comparison

### Real-Time Prediction
- Input transaction features
- Get predictions from all 3 models with confidence scores

### Data Insights
- SMOTE before/after visualization
- Feature importance from Random Forest
- Correlation heatmap

---

## 📖 How It Works

1. **Data Preprocessing** — Scale `Time` and `Amount` features using StandardScaler
2. **Train/Test Split** — 80/20 stratified split to maintain class distribution
3. **SMOTE** — Generate synthetic fraud samples in training data only (prevents data leakage)
4. **Model Training** — Train Logistic Regression, Random Forest, and Decision Tree
5. **Evaluation** — Compare using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
6. **Prediction** — Use trained models to classify new transactions in real-time

---

## 📝 Dataset Info

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions:** 284,807
- **Fraud Cases:** 492 (0.172%)
- **Features:** 30 (V1-V28 from PCA, Time, Amount)
- **Target:** Class (0 = Normal, 1 = Fraud)

---

## 👥 Authors

- Mini Project — Credit Card Fraud Detection

---

*Built with ❤️ using Python, Scikit-Learn & Streamlit*
