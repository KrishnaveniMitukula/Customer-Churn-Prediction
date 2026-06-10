<div align="center">

# 📊 Customer Churn Prediction System

### AI-Powered Risk Assessment & Retention Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://customer-churn-prediction-ognbl7uxzy9hpr9r8pckng.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1+-017CEE?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.46+-B721FF?style=for-the-badge)](https://shap.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

*A production-grade, end-to-end machine learning system for predicting customer churn in the telecommunications industry, featuring explainable AI, automated retention recommendations, and an interactive Streamlit dashboard.*

**Author: Krishnaveni Mitukula**

</div>

---

## 🏗️ Project Architecture

```
Customer-Churn-Prediction/
│
├── 📂 data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,043 customers)
│   └── cleaned_churn_data.csv                   # Cleaned dataset (auto-generated)
│
├── 📂 model/
│   ├── churn_model.pkl                          # Trained XGBoost classifier
│   ├── feature_columns.pkl                      # Ordered feature column list
│   ├── label_encoders.pkl                       # Fitted LabelEncoder dictionary
│   └── scaler.pkl                               # Fitted StandardScaler
│
├── 📂 images/
│   ├── churn_distribution.png                   # Churn pie chart + countplot
│   ├── categorical_drivers.png                  # Categorical features vs churn
│   ├── heatmap.png                              # Numerical correlation heatmap
│   ├── tenure_boxplot.png                       # Tenure distribution by churn
│   ├── monthlycharges_boxplot.png               # Monthly charges by churn
│   ├── totalcharges_distribution.html           # Interactive Plotly distribution
│   ├── roc_curve.png                            # ROC curve with optimal threshold
│   ├── precision_recall_curve.png               # Precision-Recall curve
│   ├── feature_importance.png                   # XGBoost feature importances
│   ├── confusion_matrix.png                     # Confusion matrix heatmap
│   ├── shap_summary.png                         # SHAP beeswarm summary plot
│   ├── shap_bar.png                             # SHAP global bar plot
│   ├── shap_waterfall.png                       # SHAP local waterfall plot
│   └── shap_force_plot.html                     # SHAP interactive force plot
│
├── 📂 reports/
│   ├── business_insights.txt                    # EDA business insights report
│   └── model_report.txt                         # Model evaluation report
│
├── 📄 eda_analysis.py                           # Phase 1-2: EDA & Data Cleaning
├── 📄 train_model.py                            # Phase 3-6: Training & Tuning
├── 📄 explainability.py                         # Phase 7-8: SHAP Explainability
├── 📄 app.py                                    # Phase 9-10: Streamlit Dashboard
├── 📄 requirements.txt                          # Pinned Python dependencies
├── 🐳 Dockerfile                                # Multi-stage Docker build
├── 🐳 docker-compose.yml                        # Docker Compose orchestration
├── 📄 .gitignore                                # Git ignore rules
├── 📄 LICENSE                                   # MIT License
└── 📄 README.md                                 # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** installed
- **pip** package manager
- **Git** (optional, for cloning)

### 1. Clone & Navigate

```bash
git clone https://github.com/krishnaveni-mitukula/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place the CSV file in the `data/` directory:

```
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 5. Run the Pipeline (Sequential)

```bash
# Step 1: Exploratory Data Analysis & Cleaning
python eda_analysis.py

# Step 2: Model Training, Tuning & Evaluation
python train_model.py

# Step 3: SHAP Explainability Analysis
python explainability.py

# Step 4: Launch the Streamlit Dashboard
streamlit run app.py
```

The dashboard will be available locally at **http://localhost:8501** (or view the live cloud deployment at [customer-churn-prediction.streamlit.app](https://customer-churn-prediction-ognbl7uxzy9hpr9r8pckng.streamlit.app/)).

---

## 🐳 Docker Deployment

### Build & Run with Docker Compose

```bash
# Build and start
docker-compose up --build

# Run in detached mode
docker-compose up --build -d

# View logs
docker-compose logs -f churn-app

# Stop
docker-compose down
```

### Build & Run with Docker CLI

```bash
# Build the image
docker build -t churn-prediction .

# Run the container
docker run -p 8501:8501 \
  -v ./data:/app/data:ro \
  -v ./model:/app/model:ro \
  churn-prediction
```

Access the dashboard locally at **http://localhost:8501** or access the live app [here](https://customer-churn-prediction-ognbl7uxzy9hpr9r8pckng.streamlit.app/).

---

## 📈 Pipeline Overview

### Phase 1–2: Exploratory Data Analysis (`eda_analysis.py`)

| Step | Description |
|------|-------------|
| **Data Loading** | Loads 7,043 customer records with 21 features |
| **Profiling** | Null analysis, dtype mapping, duplicate detection |
| **TotalCharges Cleaning** | Whitespace stripping → numeric coercion → tenure-cohort median imputation |
| **Target Encoding** | Binary mapping: `Yes → 1`, `No → 0` |
| **Visualizations** | 6 publication-quality plots (pie, countplot, heatmap, boxplots, Plotly histogram) |
| **Business Report** | 7-section structured report with churn rates, correlations, and strategic recommendations |

### Phase 3–6: Model Training (`train_model.py`)

| Step | Description |
|------|-------------|
| **Feature Engineering** | TenureGroup, MonthlyChargeCategory, CLV_Estimate, ContractRiskScore, AvgChargesPerMonth, HasMultipleServices |
| **Encoding & Scaling** | LabelEncoder for categoricals, StandardScaler for numericals |
| **Baseline Models** | Logistic Regression, Random Forest, XGBoost — all with class imbalance handling |
| **Hyperparameter Tuning** | RandomizedSearchCV (80 iterations, 5-fold stratified CV) on XGBoost |
| **Threshold Optimization** | F1-maximizing threshold sweep (0.20–0.80) |
| **Artifact Export** | Model, feature columns, encoders, scaler → `model/` directory |

### Phase 7–8: Explainability (`explainability.py`)

| Step | Description |
|------|-------------|
| **SHAP TreeExplainer** | Exact Shapley values for tree-based models |
| **Global Plots** | Summary beeswarm + mean \|SHAP\| bar chart |
| **Local Plots** | Waterfall + Force plot for individual customer explanations |
| **`explain_customer()` API** | Programmatic interface returning Top 5 churn drivers + Top 5 retention factors |

### Phase 9–10: Dashboard (`app.py`)

| Step | Description |
|------|-------------|
| **Modern UI** | Custom CSS with glassmorphism, gradients, Inter font, animated cards |
| **Input Panel** | 17 sidebar controls covering all training features |
| **Risk Engine** | Tri-tier classification: Low (🟢) / Medium (🟠) / High (🔴) |
| **Gauge Chart** | Plotly radial gauge with color zones and threshold marker |
| **SHAP Waterfall** | Real-time per-customer feature contribution chart |
| **Retention Engine** | 8 dynamic recommendation rules mapped to customer vulnerabilities |
| **PDF Export** | Branded multi-section PDF report via download button |

---

## 🔬 Model Performance

The model is tuned using a 5-fold Stratified Cross-Validation Randomized Search on the XGBoost classifier. 

### Best Tuned Parameters
* `colsample_bytree`: `0.5061`
* `gamma`: `4.8494`
* `learning_rate`: `0.0225`
* `max_depth`: `7`
* `min_child_weight`: `6`
* `n_estimators`: `312`
* `subsample`: `0.8215`

### Evaluation Metrics (at Optimal Threshold of 0.59)
To optimize for business intervention, the decision threshold was swept to maximize the F1-Score:

| Metric | Baseline XGBoost | Tuned XGBoost (Optimal Threshold) |
|--------|------------------|------------------------------------|
| **Accuracy** | 75.66% | **78.42%** |
| **Precision** | 53.36% | **57.35%** |
| **Recall** | 65.78% | **72.99%** |
| **F1-Score** | 58.92% | **64.24%** |
| **ROC-AUC** | 0.8250 | **0.8470** |

Detailed evaluation metrics and plots are saved to the [model_report.txt](file:///c:/Users/KRISH/Downloads/Customer-Churn-Prediction/reports/model_report.txt) and `images/` directories.

---

## 🛡️ Key Design Decisions

1. **Tenure-Cohort Imputation**: TotalCharges missing values are imputed using the median within each tenure cohort, preserving the natural relationship between tenure and cumulative charges.

2. **Feature Engineering**: Six engineered features capture domain-specific patterns — CLV approximation, contract risk scoring, and service aggregation improve model discrimination.

3. **Threshold Optimization**: The default 0.50 threshold is replaced with an F1-maximizing threshold, balancing precision and recall for the business use case where both false positives and false negatives carry costs.

4. **SHAP over LIME**: TreeExplainer provides exact (not approximate) Shapley values for tree-based models, ensuring faithful explanations.

5. **Multi-Stage Docker Build**: Separates compilation (builder) from runtime, reducing final image size by ~40%.

---

## 📋 Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11+ |
| **ML Framework** | Scikit-Learn 1.5, XGBoost 2.1 |
| **Explainability** | SHAP 0.46 |
| **Dashboard** | Streamlit 1.36 |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **PDF Generation** | FPDF2 |
| **Containerization** | Docker, Docker Compose |

---

## 👩‍💻 Author

**Krishnaveni Mitukula**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by Krishnaveni Mitukula**

*Powered by XGBoost, SHAP & Streamlit*

</div>
