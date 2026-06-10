"""
explainability.py
================================================================================
Telco Customer Churn — Explainable AI (XAI) Module
================================================================================

Author  : Krishnaveni Mitukula
Version : 1.0.0
Python  : 3.11+

Description
-----------
Production-grade explainability module that:
  1. Loads the serialized XGBoost model and preprocessing artifacts.
  2. Initializes a SHAP ``TreeExplainer`` for fast, exact Shapley value
     computation on tree-based models.
  3. Generates and saves publication-quality SHAP visualizations:
     - Summary Beeswarm Plot
     - Global Bar Plot (mean |SHAP|)
     - Local Waterfall Plot (single customer)
     - Local Force Plot (single customer, HTML)
  4. Exposes ``explain_customer(customer_vector)`` — a clean utility function
     that computes local SHAP values and extracts the Top 5 churn-driving
     features and Top 5 retention-stabilizing features.

Usage
-----
    # Standalone execution (generates global plots from test data)
    python explainability.py

    # Programmatic usage from app.py or notebooks
    from explainability import load_explainability_artifacts, explain_customer
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
DATA_DIR: Path = BASE_DIR / "data"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
})


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING
# ══════════════════════════════════════════════════════════════════════════════


def load_pickle_artifact(filepath: Path, description: str) -> Any:
    """Load a single pickled artifact from disk.

    Args:
        filepath: Absolute or relative path to the ``.pkl`` file.
        description: Human-readable description for logging.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If the artifact file does not exist.
    """
    try:
        if not filepath.exists():
            raise FileNotFoundError(
                f"{description} not found at '{filepath}'. "
                "Run 'python train_model.py' first."
            )
        with open(filepath, "rb") as f:
            artifact: Any = pickle.load(f)
        file_size_kb: float = filepath.stat().st_size / 1024
        logger.info("Loaded %s from %s (%.1f KB)", description, filepath, file_size_kb)
        print(f"[INFO] Loaded {description} ({file_size_kb:.1f} KB)")
        return artifact
    except FileNotFoundError as exc:
        logger.error("FileNotFoundError: %s", exc)
        raise
    except Exception as exc:
        logger.error("Failed to load %s: %s", description, exc)
        raise


def load_explainability_artifacts() -> Tuple[Any, List[str], Dict[str, Any], Any]:
    """Load all model artifacts required for SHAP explainability.

    Returns:
        Tuple of:
          - Trained model (XGBoost classifier).
          - Feature column names (list of str).
          - Label encoders (dict mapping column → LabelEncoder).
          - Fitted StandardScaler.

    Raises:
        FileNotFoundError: If any artifact file is missing.
    """
    try:
        logger.info("Loading explainability artifacts ...")
        print("\n[INFO] Loading model artifacts for explainability ...")

        model: Any = load_pickle_artifact(
            MODEL_DIR / "churn_model.pkl", "Churn Model"
        )
        feature_columns: List[str] = load_pickle_artifact(
            MODEL_DIR / "feature_columns.pkl", "Feature Columns"
        )
        label_encoders: Dict[str, Any] = load_pickle_artifact(
            MODEL_DIR / "label_encoders.pkl", "Label Encoders"
        )
        scaler: Any = load_pickle_artifact(
            MODEL_DIR / "scaler.pkl", "Standard Scaler"
        )

        logger.info(
            "All artifacts loaded — Model type: %s | Features: %d",
            type(model).__name__,
            len(feature_columns),
        )
        print(f"[INFO] Model type: {type(model).__name__} | Features: {len(feature_columns)}")
        return model, feature_columns, label_encoders, scaler
    except Exception as exc:
        logger.error("Failed to load explainability artifacts: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# SHAP EXPLAINER INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════


def initialize_shap_explainer(model: Any) -> shap.TreeExplainer:
    """Initialize a SHAP TreeExplainer for the trained model.

    Args:
        model: Trained tree-based model (XGBoost, LightGBM, RandomForest, etc.).

    Returns:
        Initialized ``shap.TreeExplainer`` instance.
    """
    try:
        logger.info("Initializing SHAP TreeExplainer ...")
        explainer: shap.TreeExplainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized successfully.")
        print("[INFO] SHAP TreeExplainer initialized.")
        return explainer
    except Exception as exc:
        logger.error("Failed to initialize SHAP TreeExplainer: %s", exc)
        raise


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> shap.Explanation:
    """Compute SHAP values for a given dataset.

    Args:
        explainer: Initialized SHAP TreeExplainer.
        X: Feature DataFrame to explain.

    Returns:
        ``shap.Explanation`` object containing SHAP values, base values,
        feature names, and data.
    """
    try:
        logger.info("Computing SHAP values for %d samples ...", len(X))
        print(f"[INFO] Computing SHAP values for {len(X):,} samples ...")

        shap_values: shap.Explanation = explainer(X)

        logger.info("SHAP values computed — Shape: %s", shap_values.shape)
        print(f"[INFO] SHAP values computed — Shape: {shap_values.shape}")
        return shap_values
    except Exception as exc:
        logger.error("Failed to compute SHAP values: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# SHAP VISUALIZATION — GLOBAL PLOTS
# ══════════════════════════════════════════════════════════════════════════════


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    max_display: int = 20,
) -> None:
    """Generate and save the SHAP summary beeswarm plot.

    The beeswarm plot shows the distribution of SHAP values for each
    feature across all samples, with color indicating feature value.

    Args:
        shap_values: Computed SHAP Explanation object.
        X: Feature DataFrame used for computation.
        max_display: Maximum number of features to display.
    """
    try:
        logger.info("Generating SHAP summary beeswarm plot ...")
        fig, ax = plt.subplots(figsize=(14, 10))

        shap.summary_plot(
            shap_values.values,
            X,
            feature_names=X.columns.tolist(),
            max_display=max_display,
            show=False,
            plot_size=None,
        )

        plt.title(
            "SHAP Summary Plot — Feature Impact on Churn Prediction",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=13)
        plt.tight_layout()

        save_path: Path = IMAGE_DIR / "shap_summary.png"
        plt.savefig(save_path, bbox_inches="tight", facecolor="white", dpi=200)
        plt.close("all")

        logger.info("Saved SHAP summary plot: %s", save_path)
        print(f"[SAVED] SHAP summary beeswarm plot → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate SHAP summary plot: %s", exc)
        raise


def plot_shap_bar(
    shap_values: shap.Explanation,
    max_display: int = 20,
) -> None:
    """Generate and save the SHAP global bar plot (mean absolute SHAP values).

    Args:
        shap_values: Computed SHAP Explanation object.
        max_display: Maximum number of features to display.
    """
    try:
        logger.info("Generating SHAP global bar plot ...")
        fig, ax = plt.subplots(figsize=(12, 8))

        shap.plots.bar(
            shap_values,
            max_display=max_display,
            show=False,
        )

        plt.title(
            "SHAP Feature Importance — Mean |SHAP Value|",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Mean |SHAP Value|", fontsize=13)
        plt.tight_layout()

        save_path: Path = IMAGE_DIR / "shap_bar.png"
        plt.savefig(save_path, bbox_inches="tight", facecolor="white", dpi=200)
        plt.close("all")

        logger.info("Saved SHAP bar plot: %s", save_path)
        print(f"[SAVED] SHAP global bar plot → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate SHAP bar plot: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# SHAP VISUALIZATION — LOCAL PLOTS (SINGLE CUSTOMER)
# ══════════════════════════════════════════════════════════════════════════════


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    sample_index: int = 0,
    max_display: int = 15,
) -> None:
    """Generate and save a SHAP waterfall plot for a single customer.

    Args:
        shap_values: Computed SHAP Explanation object.
        sample_index: Index of the sample to explain.
        max_display: Maximum number of features to display.
    """
    try:
        logger.info("Generating SHAP waterfall plot for sample index %d ...", sample_index)

        if sample_index >= len(shap_values):
            logger.warning(
                "Sample index %d exceeds dataset size %d — using index 0.",
                sample_index, len(shap_values),
            )
            sample_index = 0

        fig, ax = plt.subplots(figsize=(12, 8))

        shap.plots.waterfall(
            shap_values[sample_index],
            max_display=max_display,
            show=False,
        )

        plt.title(
            f"SHAP Waterfall Plot — Customer #{sample_index} (Local Explanation)",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()

        save_path: Path = IMAGE_DIR / "shap_waterfall.png"
        plt.savefig(save_path, bbox_inches="tight", facecolor="white", dpi=200)
        plt.close("all")

        logger.info("Saved SHAP waterfall plot: %s", save_path)
        print(f"[SAVED] SHAP waterfall plot (sample #{sample_index}) → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate SHAP waterfall plot: %s", exc)
        raise


def plot_shap_force(
    explainer: shap.TreeExplainer,
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    sample_index: int = 0,
) -> None:
    """Generate and save a SHAP force plot for a single customer as HTML.

    Args:
        explainer: Initialized SHAP TreeExplainer.
        shap_values: Computed SHAP Explanation object.
        X: Feature DataFrame used for computation.
        sample_index: Index of the sample to explain.
    """
    try:
        logger.info("Generating SHAP force plot for sample index %d ...", sample_index)

        if sample_index >= len(X):
            logger.warning(
                "Sample index %d exceeds dataset size %d — using index 0.",
                sample_index, len(X),
            )
            sample_index = 0

        # Initialize JS visualization support
        shap.initjs()

        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values.values[sample_index],
            features=X.iloc[sample_index],
            feature_names=X.columns.tolist(),
            matplotlib=False,
        )

        # Save as interactive HTML
        html_path: Path = IMAGE_DIR / "shap_force_plot.html"
        shap.save_html(str(html_path), force_plot)
        logger.info("Saved SHAP force plot (HTML): %s", html_path)
        print(f"[SAVED] SHAP force plot (HTML) → {html_path}")

        # Also generate a static matplotlib version
        try:
            fig, ax = plt.subplots(figsize=(20, 4))
            shap.force_plot(
                base_value=explainer.expected_value,
                shap_values=shap_values.values[sample_index],
                features=X.iloc[sample_index],
                feature_names=X.columns.tolist(),
                matplotlib=True,
                show=False,
            )
            plt.title(
                f"SHAP Force Plot — Customer #{sample_index}",
                fontsize=14,
                fontweight="bold",
                pad=30,
            )
            plt.tight_layout()

            static_path: Path = IMAGE_DIR / "shap_force_plot.png"
            plt.savefig(static_path, bbox_inches="tight", facecolor="white", dpi=200)
            plt.close("all")
            logger.info("Saved SHAP force plot (static): %s", static_path)
            print(f"[SAVED] SHAP force plot (static PNG) → {static_path}")
        except Exception as static_exc:
            logger.warning("Static force plot generation failed: %s", static_exc)
            print(f"[WARN] Static force plot skipped: {static_exc}")

    except Exception as exc:
        logger.error("Failed to generate SHAP force plot: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOMER-LEVEL EXPLANATION UTILITY
# ══════════════════════════════════════════════════════════════════════════════


def explain_customer(
    customer_vector: pd.DataFrame,
    model: Any,
    feature_columns: List[str],
    top_n: int = 5,
) -> Dict[str, Any]:
    """Compute local SHAP explanations for a single customer.

    This is the primary programmatic interface for the Streamlit dashboard.
    It computes SHAP values for a single customer and extracts the
    top features driving churn risk and the top features stabilizing retention.

    Args:
        customer_vector: Single-row DataFrame with feature values matching
            the training feature columns. Must have shape ``(1, n_features)``.
        model: Trained tree-based classifier.
        feature_columns: Ordered list of feature column names used during training.
        top_n: Number of top features to extract for each direction.

    Returns:
        Dictionary containing:
          - ``shap_values``: Raw SHAP values array for the customer.
          - ``base_value``: Expected value (base rate) from the explainer.
          - ``prediction_proba``: Model's predicted churn probability.
          - ``churn_drivers``: List of dicts with top features pushing toward churn
            (positive SHAP). Each dict has ``feature``, ``shap_value``, ``feature_value``.
          - ``retention_factors``: List of dicts with top features pushing toward
            retention (negative SHAP). Each dict has ``feature``, ``shap_value``,
            ``feature_value``.
          - ``shap_explanation``: Full ``shap.Explanation`` object for plotting.

    Raises:
        ValueError: If customer_vector shape is invalid.
    """
    try:
        logger.info("Generating SHAP explanation for single customer ...")

        # ── Validate Input ───────────────────────────────────────────────────
        if customer_vector.shape[0] != 1:
            raise ValueError(
                f"Expected single-row DataFrame, got shape {customer_vector.shape}. "
                "Pass exactly one customer at a time."
            )

        # Ensure column order matches training
        customer_aligned: pd.DataFrame = customer_vector[feature_columns].copy()

        # ── Initialize Explainer & Compute ───────────────────────────────────
        explainer: shap.TreeExplainer = shap.TreeExplainer(model)
        shap_explanation: shap.Explanation = explainer(customer_aligned)

        shap_vals: np.ndarray = shap_explanation.values[0]
        base_value: float = float(explainer.expected_value)

        # ── Get Prediction Probability ───────────────────────────────────────
        prediction_proba: float = float(model.predict_proba(customer_aligned)[0, 1])

        # ── Extract Top Churn Drivers (Positive SHAP) ────────────────────────
        feature_shap_pairs: List[Tuple[str, float, Any]] = []
        for idx, feat_name in enumerate(feature_columns):
            feat_val: Any = customer_aligned.iloc[0, idx]
            feature_shap_pairs.append((feat_name, float(shap_vals[idx]), feat_val))

        # Sort by SHAP value descending for churn drivers
        sorted_positive: List[Tuple[str, float, Any]] = sorted(
            [(f, s, v) for f, s, v in feature_shap_pairs if s > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        churn_drivers: List[Dict[str, Any]] = [
            {
                "feature": feat,
                "shap_value": round(shap_val, 4),
                "feature_value": feat_val,
            }
            for feat, shap_val, feat_val in sorted_positive[:top_n]
        ]

        # Sort by SHAP value ascending for retention factors (most negative first)
        sorted_negative: List[Tuple[str, float, Any]] = sorted(
            [(f, s, v) for f, s, v in feature_shap_pairs if s < 0],
            key=lambda x: x[1],
        )
        retention_factors: List[Dict[str, Any]] = [
            {
                "feature": feat,
                "shap_value": round(shap_val, 4),
                "feature_value": feat_val,
            }
            for feat, shap_val, feat_val in sorted_negative[:top_n]
        ]

        # ── Build Result ─────────────────────────────────────────────────────
        result: Dict[str, Any] = {
            "shap_values": shap_vals,
            "base_value": base_value,
            "prediction_proba": round(prediction_proba, 4),
            "churn_drivers": churn_drivers,
            "retention_factors": retention_factors,
            "shap_explanation": shap_explanation,
        }

        # ── Log Results ──────────────────────────────────────────────────────
        logger.info("Customer churn probability: %.4f", prediction_proba)
        logger.info("Base value: %.4f", base_value)

        print(f"\n[INFO] Customer Explanation:")
        print(f"  Churn Probability : {prediction_proba:.4f} ({prediction_proba*100:.1f}%)")
        print(f"  Base Value        : {base_value:.4f}")

        print(f"\n  Top {top_n} CHURN DRIVERS (pushing toward churn):")
        for i, driver in enumerate(churn_drivers, 1):
            print(
                f"    {i}. {driver['feature']:<30} "
                f"SHAP: +{driver['shap_value']:.4f}  "
                f"(value: {driver['feature_value']})"
            )

        print(f"\n  Top {top_n} RETENTION FACTORS (stabilizing retention):")
        for i, factor in enumerate(retention_factors, 1):
            print(
                f"    {i}. {factor['feature']:<30} "
                f"SHAP: {factor['shap_value']:.4f}  "
                f"(value: {factor['feature_value']})"
            )

        return result
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        raise
    except Exception as exc:
        logger.error("Failed to explain customer: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION SUBSET LOADER
# ══════════════════════════════════════════════════════════════════════════════


def load_evaluation_subset(
    feature_columns: List[str],
    label_encoders: Dict[str, Any],
    scaler: Any,
    sample_size: int = 500,
) -> pd.DataFrame:
    """Load and preprocess an evaluation subset from the cleaned dataset.

    Applies the same encoding and scaling transformations used during training
    to produce a feature matrix compatible with the serialized model.

    Args:
        feature_columns: Ordered list of feature column names.
        label_encoders: Dictionary of fitted LabelEncoder instances.
        scaler: Fitted StandardScaler instance.
        sample_size: Number of samples to include in the evaluation subset.

    Returns:
        Preprocessed evaluation DataFrame with correct column order.
    """
    try:
        logger.info("Loading evaluation subset (n=%d) ...", sample_size)
        cleaned_path: Path = DATA_DIR / "cleaned_churn_data.csv"

        if not cleaned_path.exists():
            raise FileNotFoundError(
                f"Cleaned dataset not found at '{cleaned_path}'. "
                "Run 'python eda_analysis.py' first."
            )

        df: pd.DataFrame = pd.read_csv(cleaned_path)
        logger.info("Loaded cleaned dataset — Shape: %s", df.shape)

        # ── Feature Engineering (must replicate train_model.py) ──────────────
        # Tenure Groups
        tenure_bins: List[int] = [-1, 12, 24, 48, 60, 73]
        tenure_labels: List[str] = [
            "0-12 months", "13-24 months", "25-48 months",
            "49-60 months", "60+ months",
        ]
        df["TenureGroup"] = pd.cut(
            df["tenure"], bins=tenure_bins, labels=tenure_labels, right=True
        ).astype(str)

        # Monthly Charge Categories
        q33: float = float(df["MonthlyCharges"].quantile(0.33))
        q66: float = float(df["MonthlyCharges"].quantile(0.66))
        df["MonthlyChargeCategory"] = df["MonthlyCharges"].apply(
            lambda x: "Low" if x <= q33 else ("Medium" if x <= q66 else "High")
        )

        # CLV Estimate
        df["CLV_Estimate"] = df["tenure"] * df["MonthlyCharges"]

        # Contract Risk Score
        contract_risk_map: Dict[str, float] = {
            "Month-to-month": 3.0, "One year": 1.5, "Two year": 0.5,
        }
        df["ContractRiskScore"] = 0.0
        if "Contract" in df.columns:
            df["ContractRiskScore"] = df["Contract"].map(contract_risk_map).fillna(1.0)
        if "PaperlessBilling" in df.columns:
            mask = df["PaperlessBilling"] == "Yes"
            df.loc[mask, "ContractRiskScore"] += 1.0

        # Average Charges Per Month
        df["AvgChargesPerMonth"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

        # Has Multiple Services
        service_cols: List[str] = [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        existing_svc: List[str] = [c for c in service_cols if c in df.columns]
        df["HasMultipleServices"] = sum(
            (df[c] == "Yes").astype(int) for c in existing_svc
        ) if existing_svc else 0

        # ── Encode Categoricals ──────────────────────────────────────────────
        for col, le in label_encoders.items():
            if col in df.columns:
                # Handle unseen labels gracefully
                known_classes: set = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x, kc=known_classes, encoder=le: (
                        encoder.transform([x])[0] if x in kc else 0
                    )
                )

        # ── Select & Order Columns ───────────────────────────────────────────
        missing_cols: List[str] = [c for c in feature_columns if c not in df.columns]
        if missing_cols:
            logger.warning("Missing columns in evaluation data: %s — filling with 0.", missing_cols)
            for mc in missing_cols:
                df[mc] = 0

        X_eval: pd.DataFrame = df[feature_columns].copy()

        # ── Scale Numericals ─────────────────────────────────────────────────
        numerical_cols: List[str] = X_eval.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols and hasattr(scaler, "mean_"):
            # Only scale columns that were in the scaler's training set
            scale_cols: List[str] = [
                c for c in numerical_cols
                if c in feature_columns[:len(scaler.mean_)]
            ]
            if scale_cols:
                try:
                    X_eval[scale_cols] = scaler.transform(X_eval[scale_cols])
                except ValueError:
                    logger.warning("Scaler transform dimension mismatch — scaling all numericals.")
                    X_eval[numerical_cols] = scaler.transform(X_eval[numerical_cols])

        # ── Sample ───────────────────────────────────────────────────────────
        actual_sample: int = min(sample_size, len(X_eval))
        X_sample: pd.DataFrame = X_eval.sample(
            n=actual_sample, random_state=42
        ).reset_index(drop=True)

        logger.info("Evaluation subset ready — Shape: %s", X_sample.shape)
        print(f"[INFO] Evaluation subset loaded: {X_sample.shape}")
        return X_sample
    except Exception as exc:
        logger.error("Failed to load evaluation subset: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point for the explainability pipeline.

    Orchestrates:
        1. Load model artifacts.
        2. Load evaluation subset.
        3. Initialize SHAP TreeExplainer.
        4. Compute SHAP values.
        5. Generate and save global plots (summary, bar).
        6. Generate and save local plots (waterfall, force) for a sample customer.
        7. Demonstrate ``explain_customer()`` utility function.
    """
    try:
        logger.info("=" * 70)
        logger.info("TELCO CUSTOMER CHURN — EXPLAINABILITY PIPELINE STARTED")
        logger.info("=" * 70)

        print("\n" + "█" * 70)
        print("  TELCO CUSTOMER CHURN — EXPLAINABLE AI (XAI)")
        print("  Author: Krishnaveni Mitukula")
        print("█" * 70 + "\n")

        # ── Step 1: Load Artifacts ───────────────────────────────────────────
        model, feature_columns, label_encoders, scaler = load_explainability_artifacts()

        # ── Step 2: Load Evaluation Subset ───────────────────────────────────
        X_eval: pd.DataFrame = load_evaluation_subset(
            feature_columns, label_encoders, scaler, sample_size=500
        )

        # ── Step 3: Initialize Explainer ─────────────────────────────────────
        explainer: shap.TreeExplainer = initialize_shap_explainer(model)

        # ── Step 4: Compute SHAP Values ──────────────────────────────────────
        shap_values: shap.Explanation = compute_shap_values(explainer, X_eval)

        # ── Step 5: Generate Global Plots ────────────────────────────────────
        print("\n" + "=" * 70)
        print("  GENERATING SHAP VISUALIZATIONS")
        print("=" * 70)

        plot_shap_summary(shap_values, X_eval, max_display=20)
        plot_shap_bar(shap_values, max_display=20)

        # ── Step 6: Generate Local Plots ─────────────────────────────────────
        # Pick a sample customer (index 0) for local explanation
        sample_idx: int = 0
        plot_shap_waterfall(shap_values, sample_index=sample_idx, max_display=15)
        plot_shap_force(explainer, shap_values, X_eval, sample_index=sample_idx)

        # ── Step 7: Demonstrate explain_customer() ───────────────────────────
        print("\n" + "=" * 70)
        print("  DEMONSTRATING explain_customer() UTILITY")
        print("=" * 70)

        sample_customer: pd.DataFrame = X_eval.iloc[[sample_idx]]
        explanation: Dict[str, Any] = explain_customer(
            customer_vector=sample_customer,
            model=model,
            feature_columns=feature_columns,
            top_n=5,
        )

        # ── Final Summary ────────────────────────────────────────────────────
        print("\n" + "█" * 70)
        print("  EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  SHAP plots saved       : images/")
        print(f"    - shap_summary.png   : Global beeswarm plot")
        print(f"    - shap_bar.png       : Global feature importance bar plot")
        print(f"    - shap_waterfall.png : Local waterfall plot (sample #{sample_idx})")
        print(f"    - shap_force_plot.*  : Local force plot (HTML + PNG)")
        print("█" * 70 + "\n")

        logger.info("Explainability pipeline completed successfully.")

    except FileNotFoundError as exc:
        logger.critical("PIPELINE ABORTED — Missing artifact: %s", exc)
        print(f"\n[CRITICAL] Pipeline aborted: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.critical("PIPELINE ABORTED — Unexpected error: %s", exc)
        print(f"\n[CRITICAL] Pipeline aborted: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
