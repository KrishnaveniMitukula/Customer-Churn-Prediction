"""
app.py
================================================================================
Telco Customer Churn — Streamlit Prediction Dashboard & Retention Engine
================================================================================

Author  : Krishnaveni Mitukula
Version : 1.0.0
Python  : 3.11+

Description
-----------
Production-grade Streamlit application that provides:
  1. A modern, full-width responsive UI with custom CSS injections (cards,
     badges, grids, gradients).
  2. Sidebar input panel replicating every training feature (selectboxes
     for categoricals, sliders for continuous metrics).
  3. Real-time churn probability prediction with risk-tiered UI banners
     (Low / Medium / High) and a Plotly gauge chart.
  4. Local SHAP waterfall chart rendered in the dashboard for per-customer
     interpretability.
  5. Retention Recommendation Engine mapping customer vulnerabilities to
     specific, actionable retention strategies.
  6. PDF report generation (via FPDF) bound to a download button, producing
     a client-facing summary with prediction details and recommendations.

Usage
-----
    streamlit run app.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import io
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from fpdf import FPDF

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(funcName)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ──────────────────────────────────────────────────────────────────────────────
# PATH CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
MODEL_DIR: Path = BASE_DIR / "model"
IMAGE_DIR: Path = BASE_DIR / "images"

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Customer Churn Predictor — Telco Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Telco Customer Churn Prediction Dashboard — Built by Krishnaveni Mitukula",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS: str = """
<style>
    /* ── Import Google Font ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ─────────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* ── Main Header ────────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 30px 40px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-top: 8px;
        font-weight: 300;
    }

    /* ── Metric Cards ───────────────────────────────────────────────────── */
    .metric-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 16px;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        border-color: rgba(255, 255, 255, 0.2);
    }
    .metric-card .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 8px 0;
        line-height: 1;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255, 255, 255, 0.6);
    }
    .metric-card .metric-sublabel {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.45);
        margin-top: 4px;
    }

    /* ── Risk Banners ───────────────────────────────────────────────────── */
    .risk-banner {
        border-radius: 14px;
        padding: 20px 28px;
        margin: 20px 0;
        font-weight: 600;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .risk-low {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #1a1a2e;
    }
    .risk-high {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }

    /* ── Section Cards ──────────────────────────────────────────────────── */
    .section-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 28px;
        margin: 16px 0;
    }
    .section-card h3 {
        color: #a78bfa;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(167, 139, 250, 0.2);
    }

    /* ── Recommendation Items ───────────────────────────────────────────── */
    .recommendation-item {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #667eea;
        border-radius: 0 10px 10px 0;
        padding: 14px 20px;
        margin: 10px 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
        line-height: 1.5;
        transition: all 0.2s ease;
    }
    .recommendation-item:hover {
        background: rgba(255, 255, 255, 0.08);
        border-left-color: #a78bfa;
    }
    .recommendation-item strong {
        color: #667eea;
    }

    /* ── Sidebar Styling ────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500;
    }

    /* ── Badge ──────────────────────────────────────────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-low { background: #00b09b; color: white; }
    .badge-medium { background: #f7971e; color: #1a1a2e; }
    .badge-high { background: #e74c3c; color: white; }

    /* ── Footer ─────────────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.3);
        font-size: 0.8rem;
        padding: 30px 0 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 40px;
    }

    /* ── Streamlit Overrides ────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (CACHED)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> Tuple[Any, List[str], Dict[str, Any], Any]:
    """Load and cache all serialized model artifacts.

    Returns:
        Tuple of (model, feature_columns, label_encoders, scaler).

    Raises:
        FileNotFoundError: If any artifact file is missing.
    """
    try:
        logger.info("Loading model artifacts ...")
        artifacts: Dict[str, Tuple[str, str]] = {
            "model": ("churn_model.pkl", "Churn Model"),
            "features": ("feature_columns.pkl", "Feature Columns"),
            "encoders": ("label_encoders.pkl", "Label Encoders"),
            "scaler": ("scaler.pkl", "Standard Scaler"),
        }

        loaded: Dict[str, Any] = {}
        for key, (filename, description) in artifacts.items():
            filepath: Path = MODEL_DIR / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"{description} not found at '{filepath}'. "
                    "Run 'python train_model.py' first."
                )
            with open(filepath, "rb") as f:
                loaded[key] = pickle.load(f)
            logger.info("Loaded %s from %s", description, filepath)

        return (
            loaded["model"],
            loaded["features"],
            loaded["encoders"],
            loaded["scaler"],
        )
    except Exception as exc:
        logger.error("Failed to load model artifacts: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CUSTOMER INPUT PANEL
# ══════════════════════════════════════════════════════════════════════════════


def render_sidebar_inputs() -> Dict[str, Any]:
    """Render the sidebar input panel and collect customer feature values.

    Returns:
        Dictionary of raw (unprocessed) customer feature values.
    """
    try:
        st.sidebar.markdown(
            """
            <div style="text-align:center; padding: 15px 0 20px;">
                <h2 style="color: #a78bfa; margin:0; font-size:1.4rem;">📋 Customer Profile</h2>
                <p style="color: rgba(255,255,255,0.5); font-size:0.85rem; margin-top:5px;">
                    Enter customer details below
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        inputs: Dict[str, Any] = {}

        # ── Demographics ─────────────────────────────────────────────────────
        st.sidebar.markdown("#### 👤 Demographics")
        inputs["gender"] = st.sidebar.selectbox(
            "Gender", options=["Male", "Female"], index=0, key="inp_gender"
        )
        inputs["SeniorCitizen"] = st.sidebar.selectbox(
            "Senior Citizen", options=["No", "Yes"], index=0, key="inp_senior"
        )
        inputs["Partner"] = st.sidebar.selectbox(
            "Partner", options=["No", "Yes"], index=0, key="inp_partner"
        )
        inputs["Dependents"] = st.sidebar.selectbox(
            "Dependents", options=["No", "Yes"], index=0, key="inp_dependents"
        )

        st.sidebar.markdown("---")

        # ── Account Information ──────────────────────────────────────────────
        st.sidebar.markdown("#### 📊 Account Information")
        inputs["tenure"] = st.sidebar.slider(
            "Tenure (months)", min_value=0, max_value=72, value=12, step=1, key="inp_tenure"
        )
        inputs["MonthlyCharges"] = st.sidebar.slider(
            "Monthly Charges ($)", min_value=18.0, max_value=120.0, value=50.0,
            step=0.5, key="inp_monthly"
        )
        inputs["TotalCharges"] = st.sidebar.slider(
            "Total Charges ($)", min_value=18.0, max_value=9000.0,
            value=float(inputs["tenure"] * inputs["MonthlyCharges"]),
            step=10.0, key="inp_total"
        )
        inputs["Contract"] = st.sidebar.selectbox(
            "Contract Type",
            options=["Month-to-month", "One year", "Two year"],
            index=0, key="inp_contract"
        )
        inputs["PaperlessBilling"] = st.sidebar.selectbox(
            "Paperless Billing", options=["No", "Yes"], index=1, key="inp_paperless"
        )
        inputs["PaymentMethod"] = st.sidebar.selectbox(
            "Payment Method",
            options=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=0, key="inp_payment"
        )

        st.sidebar.markdown("---")

        # ── Phone Services ───────────────────────────────────────────────────
        st.sidebar.markdown("#### 📱 Phone Services")
        inputs["PhoneService"] = st.sidebar.selectbox(
            "Phone Service", options=["No", "Yes"], index=1, key="inp_phone"
        )
        inputs["MultipleLines"] = st.sidebar.selectbox(
            "Multiple Lines",
            options=["No", "Yes", "No phone service"],
            index=0, key="inp_multilines"
        )

        st.sidebar.markdown("---")

        # ── Internet Services ────────────────────────────────────────────────
        st.sidebar.markdown("#### 🌐 Internet Services")
        inputs["InternetService"] = st.sidebar.selectbox(
            "Internet Service",
            options=["DSL", "Fiber optic", "No"],
            index=0, key="inp_internet"
        )
        inputs["OnlineSecurity"] = st.sidebar.selectbox(
            "Online Security",
            options=["No", "Yes", "No internet service"],
            index=0, key="inp_security"
        )
        inputs["OnlineBackup"] = st.sidebar.selectbox(
            "Online Backup",
            options=["No", "Yes", "No internet service"],
            index=0, key="inp_backup"
        )
        inputs["DeviceProtection"] = st.sidebar.selectbox(
            "Device Protection",
            options=["No", "Yes", "No internet service"],
            index=0, key="inp_device"
        )
        inputs["TechSupport"] = st.sidebar.selectbox(
            "Tech Support",
            options=["No", "Yes", "No internet service"],
            index=0, key="inp_techsupport"
        )
        inputs["StreamingTV"] = st.sidebar.selectbox(
            "Streaming TV",
            options=["No", "Yes", "No internet service"],
            index=0, key="inp_streamingtv"
        )
        inputs["StreamingMovies"] = st.sidebar.selectbox(
            "Streaming Movies",
            options=["No", "Yes", "No internet service"],
            index=0, key="inp_streamingmovies"
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
            <div style="text-align:center; padding:10px; color: rgba(255,255,255,0.3); font-size:0.75rem;">
                Built by Krishnaveni Mitukula
            </div>
            """,
            unsafe_allow_html=True,
        )

        return inputs
    except Exception as exc:
        logger.error("Error rendering sidebar inputs: %s", exc)
        st.error(f"Error rendering inputs: {exc}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def preprocess_customer_input(
    raw_inputs: Dict[str, Any],
    feature_columns: List[str],
    label_encoders: Dict[str, Any],
    scaler: Any,
) -> pd.DataFrame:
    """Transform raw sidebar inputs into a model-ready feature vector.

    Applies the same feature engineering, encoding, and scaling used during
    training to ensure consistency.

    Args:
        raw_inputs: Dictionary of raw customer feature values from the sidebar.
        feature_columns: Ordered list of feature column names.
        label_encoders: Fitted LabelEncoder dictionary.
        scaler: Fitted StandardScaler.

    Returns:
        Single-row DataFrame ready for model prediction.
    """
    try:
        logger.info("Preprocessing customer input ...")

        # ── Build Base DataFrame ─────────────────────────────────────────────
        df: pd.DataFrame = pd.DataFrame([raw_inputs])

        # Convert SeniorCitizen to numeric
        df["SeniorCitizen"] = 1 if raw_inputs["SeniorCitizen"] == "Yes" else 0

        # ── Feature Engineering ──────────────────────────────────────────────
        tenure: int = int(raw_inputs["tenure"])
        monthly: float = float(raw_inputs["MonthlyCharges"])
        total: float = float(raw_inputs["TotalCharges"])

        # Tenure Group
        if tenure <= 12:
            df["TenureGroup"] = "0-12 months"
        elif tenure <= 24:
            df["TenureGroup"] = "13-24 months"
        elif tenure <= 48:
            df["TenureGroup"] = "25-48 months"
        elif tenure <= 60:
            df["TenureGroup"] = "49-60 months"
        else:
            df["TenureGroup"] = "60+ months"

        # Monthly Charge Category (using training quantile approximations)
        if monthly <= 35.0:
            df["MonthlyChargeCategory"] = "Low"
        elif monthly <= 70.0:
            df["MonthlyChargeCategory"] = "Medium"
        else:
            df["MonthlyChargeCategory"] = "High"

        # CLV Estimate
        df["CLV_Estimate"] = tenure * monthly

        # Contract Risk Score
        contract_risk_map: Dict[str, float] = {
            "Month-to-month": 3.0, "One year": 1.5, "Two year": 0.5,
        }
        risk_score: float = contract_risk_map.get(raw_inputs["Contract"], 1.0)
        if raw_inputs["PaperlessBilling"] == "Yes":
            risk_score += 1.0
        df["ContractRiskScore"] = risk_score

        # Average Charges Per Month
        df["AvgChargesPerMonth"] = total / max(tenure, 1)

        # Has Multiple Services
        service_keys: List[str] = [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        df["HasMultipleServices"] = sum(
            1 for k in service_keys if raw_inputs.get(k) == "Yes"
        )

        # ── Encode Categoricals ──────────────────────────────────────────────
        for col, le in label_encoders.items():
            if col in df.columns:
                val: str = str(df[col].iloc[0])
                known_classes: set = set(le.classes_)
                if val in known_classes:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = 0
                    logger.warning("Unknown category '%s' for column '%s' — defaulting to 0.", val, col)

        # ── Ensure All Feature Columns Present ───────────────────────────────
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0

        # ── Order & Select ───────────────────────────────────────────────────
        df = df[feature_columns].copy()

        # ── Scale Numerical Features ─────────────────────────────────────────
        numerical_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols and hasattr(scaler, "mean_"):
            try:
                df[numerical_cols] = scaler.transform(df[numerical_cols])
            except ValueError:
                logger.warning("Scaler dimension mismatch — attempting partial scaling.")
                n_scaler_features: int = len(scaler.mean_)
                scale_cols: List[str] = numerical_cols[:n_scaler_features]
                df[scale_cols] = scaler.transform(df[scale_cols])

        logger.info("Preprocessing complete — Shape: %s", df.shape)
        return df
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# RISK CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════


def classify_risk(probability: float) -> Tuple[str, str, str, str]:
    """Classify churn probability into risk tier with UI properties.

    Args:
        probability: Churn probability (0.0 to 1.0).

    Returns:
        Tuple of (risk_level, css_class, emoji, badge_class).
    """
    try:
        pct: float = probability * 100
        if pct <= 30:
            return "LOW RISK", "risk-low", "✅", "badge-low"
        elif pct <= 70:
            return "MEDIUM RISK", "risk-medium", "⚠️", "badge-medium"
        else:
            return "HIGH RISK", "risk-high", "🚨", "badge-high"
    except Exception as exc:
        logger.error("Risk classification failed: %s", exc)
        return "UNKNOWN", "risk-medium", "❓", "badge-medium"


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY GAUGE CHART
# ══════════════════════════════════════════════════════════════════════════════


def create_gauge_chart(probability: float) -> go.Figure:
    """Create a Plotly gauge chart for churn probability visualization.

    Args:
        probability: Churn probability (0.0 to 1.0).

    Returns:
        Plotly ``Figure`` object containing the gauge chart.
    """
    try:
        pct: float = round(probability * 100, 1)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=pct,
                number={
                    "suffix": "%",
                    "font": {"size": 48, "color": "white", "family": "Inter"},
                },
                delta={
                    "reference": 50,
                    "relative": False,
                    "increasing": {"color": "#e74c3c"},
                    "decreasing": {"color": "#2ecc71"},
                    "font": {"size": 16},
                },
                title={
                    "text": "Churn Probability",
                    "font": {"size": 18, "color": "rgba(255,255,255,0.7)", "family": "Inter"},
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 2,
                        "tickcolor": "rgba(255,255,255,0.3)",
                        "tickfont": {"size": 12, "color": "rgba(255,255,255,0.5)"},
                    },
                    "bar": {"color": "#667eea", "thickness": 0.3},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(46, 204, 113, 0.3)"},
                        {"range": [30, 70], "color": "rgba(241, 196, 15, 0.3)"},
                        {"range": [70, 100], "color": "rgba(231, 76, 60, 0.3)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": pct,
                    },
                },
            )
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "white", "family": "Inter"},
            height=320,
            margin=dict(l=30, r=30, t=60, b=20),
        )

        return fig
    except Exception as exc:
        logger.error("Failed to create gauge chart: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# SHAP WATERFALL CHART (STREAMLIT)
# ══════════════════════════════════════════════════════════════════════════════


def render_shap_waterfall(
    model: Any,
    customer_vector: pd.DataFrame,
    feature_columns: List[str],
) -> Optional[plt.Figure]:
    """Generate a SHAP waterfall plot for a single customer.

    Args:
        model: Trained tree-based model.
        customer_vector: Single-row DataFrame with features.
        feature_columns: Ordered list of feature column names.

    Returns:
        Matplotlib Figure object, or None on failure.
    """
    try:
        logger.info("Generating SHAP waterfall for customer ...")
        explainer: shap.TreeExplainer = shap.TreeExplainer(model)
        customer_aligned: pd.DataFrame = customer_vector[feature_columns].copy()
        shap_explanation: shap.Explanation = explainer(customer_aligned)

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        shap.plots.waterfall(
            shap_explanation[0],
            max_display=12,
            show=False,
        )

        # Force all dark text to white/light grey for contrast against dark background
        import matplotlib.text
        for text_obj in fig.findobj(match=matplotlib.text.Text):
            color = text_obj.get_color()
            is_dark = False
            if isinstance(color, str):
                if color.lower() in ("black", "k", "#000000", "#111111", "#222222", "#333333"):
                    is_dark = True
            elif isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
                if color[0] < 0.35 and color[1] < 0.35 and color[2] < 0.35:
                    is_dark = True
            
            if is_dark:
                text_obj.set_color("#e2e8f0")

        # Force dark line/grid objects to a visible muted color
        import matplotlib.lines
        for line in fig.findobj(match=matplotlib.lines.Line2D):
            color = line.get_color()
            is_dark = False
            if isinstance(color, str):
                if color.lower() in ("black", "k", "#000000", "#111111", "#222222", "#333333"):
                    is_dark = True
            elif isinstance(color, (tuple, list, np.ndarray)) and len(color) >= 3:
                if color[0] < 0.35 and color[1] < 0.35 and color[2] < 0.35:
                    is_dark = True
            
            if is_dark:
                line.set_color("#44445c")

        plt.title(
            "Feature Contributions to Churn Prediction",
            fontsize=14, fontweight="bold", color="white", pad=15,
        )
        plt.tight_layout()

        return fig
    except Exception as exc:
        logger.error("SHAP waterfall generation failed: %s", exc)
        return None


def get_shap_explanation(
    model: Any,
    customer_vector: pd.DataFrame,
    feature_columns: List[str],
    top_n: int = 5,
) -> Dict[str, Any]:
    """Compute SHAP-based explanation for a customer (lightweight wrapper).

    Args:
        model: Trained classifier.
        customer_vector: Single-row feature DataFrame.
        feature_columns: Ordered list of feature column names.
        top_n: Number of top features per direction.

    Returns:
        Dictionary with churn_drivers and retention_factors lists.
    """
    try:
        explainer: shap.TreeExplainer = shap.TreeExplainer(model)
        customer_aligned: pd.DataFrame = customer_vector[feature_columns].copy()
        shap_explanation: shap.Explanation = explainer(customer_aligned)
        shap_vals: np.ndarray = shap_explanation.values[0]

        pairs: List[Tuple[str, float]] = [
            (feature_columns[i], float(shap_vals[i]))
            for i in range(len(feature_columns))
        ]

        churn_drivers: List[Dict[str, Any]] = [
            {"feature": f, "shap_value": round(s, 4)}
            for f, s in sorted(
                [(f, s) for f, s in pairs if s > 0],
                key=lambda x: x[1], reverse=True,
            )[:top_n]
        ]

        retention_factors: List[Dict[str, Any]] = [
            {"feature": f, "shap_value": round(s, 4)}
            for f, s in sorted(
                [(f, s) for f, s in pairs if s < 0],
                key=lambda x: x[1],
            )[:top_n]
        ]

        return {
            "churn_drivers": churn_drivers,
            "retention_factors": retention_factors,
            "base_value": float(explainer.expected_value),
        }
    except Exception as exc:
        logger.error("SHAP explanation failed: %s", exc)
        return {"churn_drivers": [], "retention_factors": [], "base_value": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# RETENTION RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════


def generate_retention_recommendations(
    raw_inputs: Dict[str, Any],
    probability: float,
) -> List[Dict[str, str]]:
    """Generate dynamic retention recommendations based on customer profile.

    Maps specific customer vulnerabilities to targeted action items.

    Args:
        raw_inputs: Raw customer feature values.
        probability: Predicted churn probability.

    Returns:
        List of recommendation dicts with 'title', 'action', and 'priority'.
    """
    try:
        recommendations: List[Dict[str, str]] = []

        # ── Contract Vulnerability ───────────────────────────────────────────
        if raw_inputs.get("Contract") == "Month-to-month":
            recommendations.append({
                "title": "📋 Contract Migration",
                "action": (
                    "Customer is on a <strong>month-to-month contract</strong>, the highest churn "
                    "risk segment. Offer a <strong>15–20% discount</strong> for migrating to an "
                    "annual or two-year contract. Consider waiving early termination fees as an "
                    "additional incentive."
                ),
                "priority": "HIGH",
            })

        # ── Tech Support Gap ─────────────────────────────────────────────────
        if raw_inputs.get("TechSupport") in ("No", "No internet service"):
            recommendations.append({
                "title": "🔧 Tech Support Bundle",
                "action": (
                    "Customer currently has <strong>no tech support</strong>. Proactively offer "
                    "a <strong>complimentary 3-month tech support trial</strong> or a bundled "
                    "package at a reduced rate. Customers with tech support churn at significantly "
                    "lower rates."
                ),
                "priority": "HIGH",
            })

        # ── Electronic Check Payment ─────────────────────────────────────────
        if raw_inputs.get("PaymentMethod") == "Electronic check":
            recommendations.append({
                "title": "💳 Payment Method Optimization",
                "action": (
                    "Customer uses <strong>electronic check</strong>, the payment method with the "
                    "highest churn correlation. Incentivize switching to <strong>automatic credit "
                    "card or bank transfer</strong> with a one-time $10 account credit."
                ),
                "priority": "MEDIUM",
            })

        # ── No Online Security ───────────────────────────────────────────────
        if raw_inputs.get("OnlineSecurity") in ("No",):
            recommendations.append({
                "title": "🔒 Online Security Upsell",
                "action": (
                    "Customer lacks <strong>online security</strong> protection. Position this as "
                    "a value-add: offer a <strong>bundled security + backup package</strong> at a "
                    "discounted rate to increase service stickiness and perceived value."
                ),
                "priority": "MEDIUM",
            })

        # ── Short Tenure ─────────────────────────────────────────────────────
        if raw_inputs.get("tenure", 0) <= 12:
            recommendations.append({
                "title": "🆕 New Customer Engagement",
                "action": (
                    "Customer is in the <strong>critical first 12 months</strong> (highest churn "
                    "window). Deploy a personalized onboarding sequence: welcome call within 7 days, "
                    "usage check-in at 30 days, and a <strong>loyalty milestone reward</strong> at "
                    "the 6-month mark."
                ),
                "priority": "HIGH",
            })

        # ── Fiber Optic + High Charges ───────────────────────────────────────
        if raw_inputs.get("InternetService") == "Fiber optic" and raw_inputs.get("MonthlyCharges", 0) > 70:
            recommendations.append({
                "title": "📡 Fiber Optic Value Review",
                "action": (
                    "Customer is on <strong>Fiber optic</strong> with high monthly charges. "
                    "Conduct a <strong>competitive pricing audit</strong> and proactively offer "
                    "a loyalty pricing adjustment or speed upgrade at the same cost to reinforce "
                    "value perception."
                ),
                "priority": "MEDIUM",
            })

        # ── No Dependents / No Partner ───────────────────────────────────────
        if raw_inputs.get("Partner") == "No" and raw_inputs.get("Dependents") == "No":
            recommendations.append({
                "title": "👤 Single-User Retention",
                "action": (
                    "Customer has <strong>no partner or dependents</strong>, indicating lower "
                    "household attachment. Offer <strong>refer-a-friend rewards</strong> or "
                    "social engagement incentives to deepen brand loyalty."
                ),
                "priority": "LOW",
            })

        # ── Paperless Billing ────────────────────────────────────────────────
        if raw_inputs.get("PaperlessBilling") == "Yes":
            recommendations.append({
                "title": "📧 Billing Communication",
                "action": (
                    "Customer uses <strong>paperless billing</strong>, which correlates with "
                    "higher churn. Ensure billing communications are clear and prominent. "
                    "Add <strong>usage summaries and savings highlights</strong> to monthly "
                    "digital statements."
                ),
                "priority": "LOW",
            })

        # ── No Streaming Services ────────────────────────────────────────────
        if raw_inputs.get("StreamingTV") in ("No",) and raw_inputs.get("StreamingMovies") in ("No",):
            recommendations.append({
                "title": "🎬 Entertainment Bundle",
                "action": (
                    "Customer has <strong>no streaming services</strong>. Offer a "
                    "<strong>complimentary 1-month streaming trial</strong> to increase "
                    "service engagement and switching costs."
                ),
                "priority": "LOW",
            })

        # Sort by priority
        priority_order: Dict[str, int] = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(key=lambda r: priority_order.get(r["priority"], 3))

        logger.info("Generated %d retention recommendations.", len(recommendations))
        return recommendations
    except Exception as exc:
        logger.error("Recommendation generation failed: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════


def clean_pdf_text(text: str) -> str:
    """Clean text to be compatible with standard PDF fonts (Latin-1/CP1252)."""
    if not text:
        return ""
    # Map common Unicode characters to Latin-1 equivalents
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "-",   # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "*",   # bullet point
        "📋": "",
        "🔧": "",
        "💳": "",
        "🔒": "",
        "🆕": "",
        "📡": "",
        "👤": "",
        "📧": "",
        "🎬": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any characters that cannot be encoded in Latin-1
    return text.encode("latin-1", "ignore").decode("latin-1")


class ChurnReportPDF(FPDF):
    """Custom FPDF subclass for generating branded churn prediction reports."""

    def header(self) -> None:
        """Render the PDF header with branding."""
        self.set_fill_color(102, 126, 234)
        self.rect(0, 0, 210, 35, "F")
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(255, 255, 255)
        self.set_y(8)
        self.cell(0, 10, "Customer Churn Prediction Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, "Telco Analytics Dashboard - Confidential", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(15)

    def footer(self) -> None:
        """Render the PDF footer with page numbers."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Generated by Krishnaveni Mitukula", align="C")

    def add_section_title(self, title: str) -> None:
        """Add a styled section title.

        Args:
            title: Section title text.
        """
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(102, 126, 234)
        self.cell(0, 10, clean_pdf_text(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(102, 126, 234)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def add_key_value(self, key: str, value: str) -> None:
        """Add a key-value pair row.

        Args:
            key: Label text.
            value: Value text.
        """
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(70, 7, clean_pdf_text(key), new_x="RIGHT")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, clean_pdf_text(value), new_x="LMARGIN", new_y="NEXT")


def generate_pdf_report(
    raw_inputs: Dict[str, Any],
    probability: float,
    risk_level: str,
    recommendations: List[Dict[str, str]],
    shap_info: Dict[str, Any],
) -> bytes:
    """Generate a client-facing PDF report with prediction details.

    Args:
        raw_inputs: Raw customer feature values.
        probability: Predicted churn probability.
        risk_level: Risk tier label (LOW/MEDIUM/HIGH).
        recommendations: List of retention recommendation dicts.
        shap_info: SHAP explanation dict with churn_drivers and retention_factors.

    Returns:
        PDF content as bytes.
    """
    try:
        logger.info("Generating PDF report ...")
        pdf = ChurnReportPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)

        # ── Report Metadata ──────────────────────────────────────────────────
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(130, 130, 130)
        date_str = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        pdf.cell(0, 5, clean_pdf_text(f"Report Date: {date_str}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # ── Section 1: Prediction Summary ────────────────────────────────────
        pdf.add_section_title("1. Prediction Summary")

        pct_str: str = f"{probability * 100:.1f}%"
        pdf.add_key_value("Churn Probability:", pct_str)
        pdf.add_key_value("Risk Level:", risk_level)
        pdf.add_key_value("Model:", "XGBoost (Tuned, Optimized Threshold)")
        pdf.ln(4)

        # ── Section 2: Customer Profile ──────────────────────────────────────
        pdf.add_section_title("2. Customer Profile")

        profile_items: List[Tuple[str, str]] = [
            ("Gender:", str(raw_inputs.get("gender", "N/A"))),
            ("Senior Citizen:", str(raw_inputs.get("SeniorCitizen", "N/A"))),
            ("Partner:", str(raw_inputs.get("Partner", "N/A"))),
            ("Dependents:", str(raw_inputs.get("Dependents", "N/A"))),
            ("Tenure:", f"{raw_inputs.get('tenure', 0)} months"),
            ("Monthly Charges:", f"${raw_inputs.get('MonthlyCharges', 0):.2f}"),
            ("Total Charges:", f"${raw_inputs.get('TotalCharges', 0):.2f}"),
            ("Contract:", str(raw_inputs.get("Contract", "N/A"))),
            ("Payment Method:", str(raw_inputs.get("PaymentMethod", "N/A"))),
            ("Internet Service:", str(raw_inputs.get("InternetService", "N/A"))),
            ("Tech Support:", str(raw_inputs.get("TechSupport", "N/A"))),
            ("Online Security:", str(raw_inputs.get("OnlineSecurity", "N/A"))),
        ]

        for key, value in profile_items:
            pdf.add_key_value(key, value)
        pdf.ln(4)

        # ── Section 3: Key Risk Drivers ──────────────────────────────────────
        pdf.add_section_title("3. Key Churn Risk Drivers (SHAP Analysis)")

        churn_drivers: List[Dict[str, Any]] = shap_info.get("churn_drivers", [])
        if churn_drivers:
            for i, driver in enumerate(churn_drivers, 1):
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(200, 60, 60)
                pdf.cell(8, 7, f"{i}.")
                pdf.set_text_color(30, 30, 30)
                feature_name: str = driver.get("feature", "Unknown")
                shap_val: float = driver.get("shap_value", 0)
                pdf.cell(0, 7, clean_pdf_text(f"{feature_name}  (SHAP: +{shap_val:.4f})"), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(130, 130, 130)
            pdf.cell(0, 7, "No significant churn drivers identified.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # ── Section 4: Retention Stabilizers ─────────────────────────────────
        pdf.add_section_title("4. Retention Stabilizing Factors")

        retention_factors: List[Dict[str, Any]] = shap_info.get("retention_factors", [])
        if retention_factors:
            for i, factor in enumerate(retention_factors, 1):
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(46, 204, 113)
                pdf.cell(8, 7, f"{i}.")
                pdf.set_text_color(30, 30, 30)
                feature_name = factor.get("feature", "Unknown")
                shap_val = factor.get("shap_value", 0)
                pdf.cell(0, 7, clean_pdf_text(f"{feature_name}  (SHAP: {shap_val:.4f})"), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(130, 130, 130)
            pdf.cell(0, 7, "No significant retention factors identified.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # ── Section 5: Retention Recommendations ────────────────────────────
        pdf.add_section_title("5. Retention Action Plan")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(60, 60, 60)
                # Clean HTML tags for PDF
                title: str = rec.get("title", "").replace("<strong>", "").replace("</strong>", "")
                priority: str = rec.get("priority", "")
                pdf.cell(0, 7, clean_pdf_text(f"{i}. {title}  [{priority}]"), new_x="LMARGIN", new_y="NEXT")

                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(80, 80, 80)
                action_text: str = rec.get("action", "").replace("<strong>", "").replace("</strong>", "")
                pdf.multi_cell(0, 5, clean_pdf_text(f"   {action_text}"), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(130, 130, 130)
            pdf.cell(0, 7, "No specific recommendations at this time.", new_x="LMARGIN", new_y="NEXT")

        # ── Disclaimer ───────────────────────────────────────────────────────
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(160, 160, 160)
        pdf.multi_cell(
            0, 4,
            clean_pdf_text(
                "Disclaimer: This report is generated by an automated machine learning system. "
                "Predictions are probabilistic estimates and should be used as one input among many "
                "in customer retention decision-making. Model: XGBoost with SHAP explainability. "
                "Author: Krishnaveni Mitukula."
            ),
            new_x="LMARGIN", new_y="NEXT",
        )

        # ── Output ───────────────────────────────────────────────────────────
        pdf_bytes = pdf.output()
        if isinstance(pdf_bytes, bytearray):
            pdf_bytes = bytes(pdf_bytes)
        logger.info("PDF report generated - %d bytes", len(pdf_bytes))
        return pdf_bytes
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main Streamlit application entry point.

    Orchestrates:
        1. Header rendering.
        2. Model artifact loading.
        3. Sidebar input collection.
        4. Prediction and risk classification.
        5. Dashboard rendering (gauge, metrics, SHAP, recommendations, PDF).
    """
    try:
        # ── Header ───────────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="main-header">
                <h1>📊 Customer Churn Prediction Dashboard</h1>
                <p>AI-Powered Risk Assessment & Retention Intelligence Engine</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Load Model ──────────────────────────────────────────────────────
        try:
            model, feature_columns, label_encoders, scaler = load_model_artifacts()
        except FileNotFoundError as exc:
            st.error(
                f"⚠️ Model artifacts not found. Please run `python train_model.py` first.\n\n"
                f"Error: {exc}"
            )
            st.stop()
            return

        # ── Sidebar Inputs ───────────────────────────────────────────────────
        raw_inputs: Dict[str, Any] = render_sidebar_inputs()

        if not raw_inputs:
            st.warning("⚠️ Please configure customer profile in the sidebar.")
            st.stop()
            return

        # ── Predict Button ───────────────────────────────────────────────────
        predict_clicked: bool = st.sidebar.button(
            "🔮 Predict Churn Risk", use_container_width=True, key="btn_predict"
        )

        if predict_clicked or st.session_state.get("prediction_made", False):
            st.session_state["prediction_made"] = True
            st.session_state["raw_inputs"] = raw_inputs

            # ── Preprocess ───────────────────────────────────────────────────
            customer_vector: pd.DataFrame = preprocess_customer_input(
                raw_inputs, feature_columns, label_encoders, scaler
            )

            # ── Predict ──────────────────────────────────────────────────────
            probability: float = float(model.predict_proba(customer_vector)[0, 1])
            risk_level, risk_css, risk_emoji, badge_css = classify_risk(probability)

            # ── Risk Banner ──────────────────────────────────────────────────
            st.markdown(
                f"""
                <div class="risk-banner {risk_css}">
                    <span style="font-size: 2rem;">{risk_emoji}</span>
                    <div>
                        <div style="font-size: 1.3rem; font-weight: 700;">{risk_level}</div>
                        <div style="font-size: 0.9rem; opacity: 0.9;">
                            Churn Probability: {probability*100:.1f}% — 
                            {"Immediate intervention recommended" if probability > 0.7 else "Monitor and engage proactively" if probability > 0.3 else "Customer appears stable"}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Metric Cards Row ─────────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Churn Probability</div>
                        <div class="metric-value" style="color: {'#e74c3c' if probability > 0.7 else '#f39c12' if probability > 0.3 else '#2ecc71'};">
                            {probability*100:.1f}%
                        </div>
                        <div class="metric-sublabel">Model Confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Risk Tier</div>
                        <div class="metric-value" style="font-size: 1.6rem; color: {'#e74c3c' if 'HIGH' in risk_level else '#f39c12' if 'MEDIUM' in risk_level else '#2ecc71'};">
                            <span class="status-badge {badge_css}">{risk_level}</span>
                        </div>
                        <div class="metric-sublabel">Classification</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                tenure_val: int = raw_inputs.get("tenure", 0)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Customer Tenure</div>
                        <div class="metric-value" style="color: #a78bfa;">{tenure_val}</div>
                        <div class="metric-sublabel">Months</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col4:
                monthly_val: float = raw_inputs.get("MonthlyCharges", 0)
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Monthly Charges</div>
                        <div class="metric-value" style="color: #667eea;">${monthly_val:.0f}</div>
                        <div class="metric-sublabel">Per Month</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauge Chart + SHAP Waterfall ─────────────────────────────────
            gauge_col, shap_col = st.columns([1, 1])

            with gauge_col:
                st.markdown(
                    '<div class="section-card"><h3>🎯 Risk Probability Gauge</h3></div>',
                    unsafe_allow_html=True,
                )
                gauge_fig: go.Figure = create_gauge_chart(probability)
                st.plotly_chart(gauge_fig, use_container_width=True, key="gauge_chart")

            with shap_col:
                st.markdown(
                    '<div class="section-card"><h3>🔍 SHAP Feature Contributions</h3></div>',
                    unsafe_allow_html=True,
                )
                with st.spinner("Computing SHAP explanations ..."):
                    waterfall_fig: Optional[plt.Figure] = render_shap_waterfall(
                        model, customer_vector, feature_columns
                    )
                    if waterfall_fig is not None:
                        st.pyplot(waterfall_fig, use_container_width=True)
                        plt.close(waterfall_fig)
                    else:
                        st.info("SHAP waterfall could not be generated.")

            # ── SHAP Explanation Details ──────────────────────────────────────
            shap_info: Dict[str, Any] = get_shap_explanation(
                model, customer_vector, feature_columns, top_n=5
            )

            st.markdown("<br>", unsafe_allow_html=True)

            driver_col, factor_col = st.columns(2)

            with driver_col:
                st.markdown(
                    """
                    <div class="section-card">
                        <h3>🔴 Top Churn Risk Drivers</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for i, driver in enumerate(shap_info.get("churn_drivers", []), 1):
                    st.markdown(
                        f"""
                        <div class="recommendation-item" style="border-left-color: #e74c3c;">
                            <strong>#{i} {driver['feature']}</strong><br>
                            SHAP Impact: <span style="color: #e74c3c;">+{driver['shap_value']:.4f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with factor_col:
                st.markdown(
                    """
                    <div class="section-card">
                        <h3>🟢 Top Retention Stabilizers</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for i, factor in enumerate(shap_info.get("retention_factors", []), 1):
                    st.markdown(
                        f"""
                        <div class="recommendation-item" style="border-left-color: #2ecc71;">
                            <strong>#{i} {factor['feature']}</strong><br>
                            SHAP Impact: <span style="color: #2ecc71;">{factor['shap_value']:.4f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # ── Retention Recommendations ────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="section-card">
                    <h3>💡 Personalized Retention Action Plan</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            recommendations: List[Dict[str, str]] = generate_retention_recommendations(
                raw_inputs, probability
            )

            if recommendations:
                for rec in recommendations:
                    priority_color: str = {
                        "HIGH": "#e74c3c",
                        "MEDIUM": "#f39c12",
                        "LOW": "#3498db",
                    }.get(rec["priority"], "#667eea")

                    st.markdown(
                        f"""
                        <div class="recommendation-item" style="border-left-color: {priority_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <strong>{rec['title']}</strong>
                                <span class="status-badge" style="background: {priority_color}; color: white; font-size: 0.7rem;">
                                    {rec['priority']} PRIORITY
                                </span>
                            </div>
                            {rec['action']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ No critical vulnerabilities detected. Customer profile is stable.")

            # ── PDF Download ─────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="section-card">
                    <h3>📄 Export Customer Report</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            try:
                pdf_bytes: bytes = generate_pdf_report(
                    raw_inputs, probability, risk_level, recommendations, shap_info
                )
                timestamp_str: str = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Download Customer Report PDF",
                    data=pdf_bytes,
                    file_name=f"churn_report_{timestamp_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_download_pdf",
                )
            except Exception as pdf_exc:
                st.warning(f"⚠️ PDF generation encountered an issue: {pdf_exc}")
                logger.error("PDF generation error: %s", pdf_exc)

            # ── Customer Summary Expander ────────────────────────────────────
            with st.expander("📋 Full Customer Profile Summary", expanded=False):
                profile_df: pd.DataFrame = pd.DataFrame(
                    [
                        {"Feature": k, "Value": str(v)}
                        for k, v in raw_inputs.items()
                    ]
                )
                st.dataframe(
                    profile_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )

        else:
            # ── Welcome State ────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)

            welcome_col1, welcome_col2, welcome_col3 = st.columns(3)

            with welcome_col1:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div style="font-size: 2.5rem; margin-bottom: 10px;">🎯</div>
                        <div class="metric-label">Predict</div>
                        <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 8px;">
                            Real-time churn probability scoring powered by XGBoost
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with welcome_col2:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div style="font-size: 2.5rem; margin-bottom: 10px;">🔍</div>
                        <div class="metric-label">Explain</div>
                        <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 8px;">
                            SHAP-powered explainability for every prediction
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with welcome_col3:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div style="font-size: 2.5rem; margin-bottom: 10px;">💡</div>
                        <div class="metric-label">Retain</div>
                        <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 8px;">
                            Actionable retention strategies tailored to each customer
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="text-align:center; color: rgba(255,255,255,0.4); font-size: 1.1rem; padding: 20px;">
                    👈 Configure a customer profile in the sidebar, then click <strong>Predict Churn Risk</strong> to begin.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Footer ───────────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="footer">
                Built with ❤️ by Krishnaveni Mitukula | Powered by XGBoost, SHAP & Streamlit<br>
                © 2026 Telco Analytics — All Rights Reserved
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as exc:
        logger.critical("Application error: %s", exc)
        st.error(f"An unexpected error occurred: {exc}")


if __name__ == "__main__":
    main()
