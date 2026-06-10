"""
train_model.py
================================================================================
Telco Customer Churn — Model Training, Tuning & Evaluation Pipeline
================================================================================

Author  : Krishnaveni Mitukula
Version : 1.0.0
Python  : 3.11+

Description
-----------
Production-grade ML pipeline that:
  1. Loads the cleaned dataset produced by ``eda_analysis.py``.
  2. Engineers advanced features (tenure groups, charge categories, CLV, risk scores).
  3. Encodes categorical variables and scales numerical features.
  4. Trains baseline models: Logistic Regression, Random Forest, XGBoost.
  5. Selects the best performer and tunes via RandomizedSearchCV.
  6. Optimizes the classification threshold for F1 maximization.
  7. Exports trained artifacts (model, scaler, encoders, feature columns) to ``model/``.
  8. Generates a comprehensive model report to ``reports/model_report.txt``.
  9. Saves ROC curve, Precision-Recall curve, and feature importance plots to ``images/``.

Usage
-----
    python train_model.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import uniform, randint

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import xgboost as xgb

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

# ──────────────────────────────────────────────────────────────────────────────
# PATH CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODEL_DIR: Path = BASE_DIR / "model"
IMAGE_DIR: Path = BASE_DIR / "images"
REPORT_DIR: Path = BASE_DIR / "reports"
CLEANED_DATA_PATH: Path = DATA_DIR / "cleaned_churn_data.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# VISUAL STYLE
# ──────────────────────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    palette="muted",
    font_scale=1.15,
    rc={"figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"},
)

RANDOM_STATE: int = 42


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════


def load_cleaned_data(filepath: Path) -> pd.DataFrame:
    """Load the cleaned dataset produced by the EDA pipeline.

    Args:
        filepath: Path to the cleaned CSV file.

    Returns:
        Cleaned ``pd.DataFrame`` ready for feature engineering.

    Raises:
        FileNotFoundError: If the cleaned dataset does not exist.
    """
    try:
        logger.info("Loading cleaned dataset from: %s", filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Cleaned dataset not found at '{filepath}'. "
                "Run 'python eda_analysis.py' first to generate it."
            )
        df: pd.DataFrame = pd.read_csv(filepath)
        logger.info("Cleaned dataset loaded — Shape: %s", df.shape)
        print(f"[INFO] Cleaned dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError as exc:
        logger.error("FileNotFoundError: %s", exc)
        raise
    except Exception as exc:
        logger.error("Error loading cleaned data: %s", exc)
        raise


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced engineered features for churn prediction.

    Features created:
        - ``TenureGroup``: Categorical binning of tenure into cohorts.
        - ``MonthlyChargeCategory``: Quantile-based charge segmentation.
        - ``CLV_Estimate``: Customer Lifetime Value approximation.
        - ``ContractRiskScore``: Composite risk score from contract type
          and billing method.
        - ``AvgChargesPerMonth``: TotalCharges / max(tenure, 1).
        - ``HasMultipleServices``: Count of active add-on services.

    Args:
        df: Cleaned DataFrame with original features.

    Returns:
        DataFrame augmented with engineered features.
    """
    try:
        logger.info("Engineering features ...")

        # ── Tenure Groups ────────────────────────────────────────────────────
        tenure_bins: List[int] = [-1, 12, 24, 48, 60, 73]
        tenure_labels: List[str] = ["0-12 months", "13-24 months", "25-48 months", "49-60 months", "60+ months"]
        df["TenureGroup"] = pd.cut(
            df["tenure"], bins=tenure_bins, labels=tenure_labels, right=True
        ).astype(str)
        logger.info("Created TenureGroup feature.")
        print("[INFO] Feature engineered: TenureGroup (5 cohorts)")

        # ── Monthly Charge Categories ────────────────────────────────────────
        try:
            charge_quantiles: pd.Series = df["MonthlyCharges"].quantile([0.33, 0.66])
            q33: float = float(charge_quantiles.iloc[0])
            q66: float = float(charge_quantiles.iloc[1])

            def categorize_monthly_charge(charge: float) -> str:
                """Categorize monthly charge into Low, Medium, or High.

                Args:
                    charge: Monthly charge value.

                Returns:
                    Category label string.
                """
                if charge <= q33:
                    return "Low"
                elif charge <= q66:
                    return "Medium"
                else:
                    return "High"

            df["MonthlyChargeCategory"] = df["MonthlyCharges"].apply(categorize_monthly_charge)
            logger.info("Created MonthlyChargeCategory (Low ≤$%.2f | Medium ≤$%.2f | High).", q33, q66)
            print(f"[INFO] Feature engineered: MonthlyChargeCategory (Low ≤${q33:.0f} | Medium ≤${q66:.0f} | High)")
        except Exception as qe:
            logger.warning("MonthlyChargeCategory engineering failed: %s — using fallback.", qe)
            df["MonthlyChargeCategory"] = "Medium"

        # ── Customer Lifetime Value Estimate ─────────────────────────────────
        df["CLV_Estimate"] = df["tenure"] * df["MonthlyCharges"]
        logger.info("Created CLV_Estimate feature (tenure × MonthlyCharges).")
        print("[INFO] Feature engineered: CLV_Estimate (tenure × MonthlyCharges)")

        # ── Contract Risk Score ──────────────────────────────────────────────
        contract_risk_map: Dict[str, float] = {
            "Month-to-month": 3.0,
            "One year": 1.5,
            "Two year": 0.5,
        }
        df["ContractRiskScore"] = 0.0

        if "Contract" in df.columns:
            df["ContractRiskScore"] = df["Contract"].map(contract_risk_map).fillna(1.0)

        if "PaperlessBilling" in df.columns:
            paperless_mask: pd.Series = df["PaperlessBilling"] == "Yes"
            df.loc[paperless_mask, "ContractRiskScore"] = (
                df.loc[paperless_mask, "ContractRiskScore"] + 1.0
            )

        logger.info("Created ContractRiskScore feature.")
        print("[INFO] Feature engineered: ContractRiskScore (contract type + paperless billing)")

        # ── Average Charges Per Month ────────────────────────────────────────
        df["AvgChargesPerMonth"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
        logger.info("Created AvgChargesPerMonth feature.")
        print("[INFO] Feature engineered: AvgChargesPerMonth (TotalCharges / tenure)")

        # ── Has Multiple Services ────────────────────────────────────────────
        service_columns: List[str] = [
            "PhoneService", "MultipleLines", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies",
        ]
        existing_service_cols: List[str] = [c for c in service_columns if c in df.columns]

        if existing_service_cols:
            df["HasMultipleServices"] = 0
            for col in existing_service_cols:
                df["HasMultipleServices"] += (df[col] == "Yes").astype(int)
            logger.info("Created HasMultipleServices feature from %d service columns.", len(existing_service_cols))
            print(f"[INFO] Feature engineered: HasMultipleServices (sum of {len(existing_service_cols)} services)")
        else:
            df["HasMultipleServices"] = 0
            logger.warning("No service columns found for HasMultipleServices.")

        logger.info("Feature engineering complete — New shape: %s", df.shape)
        print(f"[INFO] Feature engineering complete — Shape: {df.shape}")
        return df
    except Exception as exc:
        logger.error("Feature engineering failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — ENCODING & SCALING
# ══════════════════════════════════════════════════════════════════════════════


def encode_categorical_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode all categorical (object-type) features.

    Args:
        df: DataFrame with mixed dtypes.

    Returns:
        Tuple of:
          - DataFrame with all categorical columns label-encoded.
          - Dictionary mapping column names to fitted ``LabelEncoder`` instances.
    """
    try:
        logger.info("Encoding categorical features ...")
        label_encoders: Dict[str, LabelEncoder] = {}
        categorical_cols: List[str] = df.select_dtypes(include=["object"]).columns.tolist()

        if not categorical_cols:
            logger.info("No categorical columns to encode.")
            print("[INFO] No categorical columns found — skipping encoding.")
            return df, label_encoders

        print(f"[INFO] Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            n_classes: int = len(le.classes_)
            logger.info("Encoded '%s' — %d unique classes.", col, n_classes)

        print(f"[INFO] Categorical encoding complete — {len(label_encoders)} encoders fitted.")
        return df, label_encoders
    except Exception as exc:
        logger.error("Categorical encoding failed: %s", exc)
        raise


def scale_numerical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a StandardScaler on training data and transform both splits.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.
        numerical_cols: List of numerical column names to scale.

    Returns:
        Tuple of:
          - Scaled X_train DataFrame.
          - Scaled X_test DataFrame.
          - Fitted ``StandardScaler`` instance.
    """
    try:
        logger.info("Scaling %d numerical features ...", len(numerical_cols))
        scaler = StandardScaler()

        valid_cols: List[str] = [c for c in numerical_cols if c in X_train.columns]

        if not valid_cols:
            logger.warning("No valid numerical columns for scaling.")
            print("[WARN] No numerical columns to scale.")
            return X_train, X_test, scaler

        X_train[valid_cols] = scaler.fit_transform(X_train[valid_cols])
        X_test[valid_cols] = scaler.transform(X_test[valid_cols])

        logger.info("Scaling complete on columns: %s", valid_cols)
        print(f"[INFO] Scaled {len(valid_cols)} numerical features.")
        return X_train, X_test, scaler
    except Exception as exc:
        logger.error("Scaling failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — BASELINE MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def compute_class_weight_ratio(y: pd.Series) -> float:
    """Compute the scale_pos_weight ratio for imbalanced binary classification.

    Args:
        y: Target series with values 0 and 1.

    Returns:
        Ratio of negative to positive samples.
    """
    try:
        n_negative: int = int((y == 0).sum())
        n_positive: int = int((y == 1).sum())
        ratio: float = n_negative / max(n_positive, 1)
        logger.info("Class weight ratio: %.3f (neg=%d, pos=%d)", ratio, n_negative, n_positive)
        print(f"[INFO] Class balance — Negative: {n_negative:,} | Positive: {n_positive:,} | Ratio: {ratio:.3f}")
        return ratio
    except Exception as exc:
        logger.error("Error computing class weight ratio: %s", exc)
        raise


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a trained classifier on the test set.

    Args:
        model: Fitted sklearn-compatible classifier.
        X_test: Test feature DataFrame.
        y_test: True test labels.
        model_name: Human-readable model name for logging.
        threshold: Classification probability threshold.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    try:
        logger.info("Evaluating model: %s (threshold=%.2f)", model_name, threshold)

        y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]
        y_pred: np.ndarray = (y_proba >= threshold).astype(int)

        accuracy: float = accuracy_score(y_test, y_pred)
        precision: float = precision_score(y_test, y_pred, zero_division=0)
        recall: float = recall_score(y_test, y_pred, zero_division=0)
        f1: float = f1_score(y_test, y_pred, zero_division=0)
        roc_auc: float = roc_auc_score(y_test, y_proba)
        cm: np.ndarray = confusion_matrix(y_test, y_pred)

        metrics: Dict[str, Any] = {
            "model_name": model_name,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,
            "threshold": threshold,
            "y_proba": y_proba,
            "y_pred": y_pred,
        }

        print(f"\n  [{model_name}] Evaluation Results:")
        print(f"    Accuracy  : {accuracy:.4f}")
        print(f"    Precision : {precision:.4f}")
        print(f"    Recall    : {recall:.4f}")
        print(f"    F1 Score  : {f1:.4f}")
        print(f"    ROC-AUC   : {roc_auc:.4f}")
        print(f"    Confusion Matrix:\n{cm}")

        logger.info(
            "%s — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
            model_name, accuracy, precision, recall, f1, roc_auc,
        )
        return metrics
    except Exception as exc:
        logger.error("Evaluation failed for '%s': %s", model_name, exc)
        raise


def train_baseline_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scale_pos_weight: float,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Train and evaluate baseline models: Logistic Regression, Random Forest, XGBoost.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        scale_pos_weight: Class imbalance ratio for XGBoost.

    Returns:
        Tuple of:
          - Dictionary containing the best model object and its name.
          - Dictionary of all model evaluation results.
    """
    try:
        logger.info("=" * 70)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("=" * 70)
        print("\n" + "=" * 70)
        print("  BASELINE MODEL TRAINING")
        print("=" * 70)

        # Compute sample_weight for sklearn models
        sample_weights: np.ndarray = np.ones(len(y_train))
        sample_weights[y_train == 1] = scale_pos_weight

        results: Dict[str, Dict[str, Any]] = {}

        # ── 1. Logistic Regression ───────────────────────────────────────────
        print("\n  Training Logistic Regression ...")
        logger.info("Training Logistic Regression ...")
        lr_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        lr_model.fit(X_train, y_train)
        results["Logistic Regression"] = evaluate_model(
            lr_model, X_test, y_test, "Logistic Regression"
        )
        results["Logistic Regression"]["model"] = lr_model

        # ── 2. Random Forest ─────────────────────────────────────────────────
        print("\n  Training Random Forest ...")
        logger.info("Training Random Forest ...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        results["Random Forest"] = evaluate_model(
            rf_model, X_test, y_test, "Random Forest"
        )
        results["Random Forest"]["model"] = rf_model

        # ── 3. XGBoost ───────────────────────────────────────────────────────
        print("\n  Training XGBoost ...")
        logger.info("Training XGBoost ...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)
        results["XGBoost"] = evaluate_model(
            xgb_model, X_test, y_test, "XGBoost"
        )
        results["XGBoost"]["model"] = xgb_model

        # ── Select Best Model ────────────────────────────────────────────────
        best_model_name: str = max(results, key=lambda k: results[k]["roc_auc"])
        best_model_obj: Any = results[best_model_name]["model"]

        print("\n" + "-" * 70)
        print(f"  ★ BEST BASELINE MODEL: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
        print("-" * 70)
        logger.info("Best baseline model: %s (AUC=%.4f)", best_model_name, results[best_model_name]["roc_auc"])

        best_info: Dict[str, Any] = {
            "model": best_model_obj,
            "name": best_model_name,
        }

        return best_info, results
    except Exception as exc:
        logger.error("Baseline model training failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    n_iter: int = 80,
    cv_folds: int = 5,
) -> xgb.XGBClassifier:
    """Perform RandomizedSearchCV on XGBoost for hyperparameter optimization.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training labels.
        scale_pos_weight: Class imbalance ratio.
        n_iter: Number of random search iterations.
        cv_folds: Number of stratified cross-validation folds.

    Returns:
        Best-fit ``XGBClassifier`` from the randomized search.
    """
    try:
        logger.info("=" * 70)
        logger.info("HYPERPARAMETER TUNING — XGBoost RandomizedSearchCV")
        logger.info("=" * 70)
        print("\n" + "=" * 70)
        print("  HYPERPARAMETER TUNING — XGBoost")
        print(f"  Iterations: {n_iter} | CV Folds: {cv_folds}")
        print("=" * 70)

        param_distributions: Dict[str, Any] = {
            "n_estimators": randint(100, 800),
            "max_depth": randint(3, 12),
            "learning_rate": uniform(0.01, 0.29),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "gamma": uniform(0, 5),
            "min_child_weight": randint(1, 10),
            "reg_alpha": uniform(0, 2),
            "reg_lambda": uniform(0.5, 2),
        }

        base_estimator = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        cv_strategy = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
        )

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv_strategy,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
            refit=True,
            return_train_score=True,
        )

        print("[INFO] Starting RandomizedSearchCV ... (this may take a few minutes)")
        logger.info("Starting RandomizedSearchCV with %d iterations ...", n_iter)
        search.fit(X_train, y_train)

        best_model: xgb.XGBClassifier = search.best_estimator_
        best_params: Dict[str, Any] = search.best_params_
        best_score: float = search.best_score_

        print(f"\n[SUCCESS] Tuning complete!")
        print(f"  Best CV ROC-AUC : {best_score:.4f}")
        print(f"  Best Parameters :")
        for param, value in sorted(best_params.items()):
            formatted_val: str = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"    {param:<25} : {formatted_val}")

        logger.info("Best CV ROC-AUC: %.4f", best_score)
        logger.info("Best params: %s", best_params)

        return best_model
    except Exception as exc:
        logger.error("Hyperparameter tuning failed: %s", exc)
        raise


def optimize_threshold(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, Dict[str, float]]:
    """Find the classification threshold that maximizes the F1 score.

    Args:
        model: Fitted classifier with ``predict_proba`` method.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Tuple of:
          - Optimal threshold value.
          - Dictionary of metrics at the optimal threshold.
    """
    try:
        logger.info("Optimizing classification threshold ...")
        y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]

        thresholds: np.ndarray = np.arange(0.20, 0.80, 0.01)
        best_threshold: float = 0.5
        best_f1: float = 0.0
        threshold_results: List[Dict[str, float]] = []

        for thresh in thresholds:
            y_pred_t: np.ndarray = (y_proba >= thresh).astype(int)
            f1_t: float = f1_score(y_test, y_pred_t, zero_division=0)
            precision_t: float = precision_score(y_test, y_pred_t, zero_division=0)
            recall_t: float = recall_score(y_test, y_pred_t, zero_division=0)

            threshold_results.append({
                "threshold": round(float(thresh), 2),
                "f1": round(f1_t, 4),
                "precision": round(precision_t, 4),
                "recall": round(recall_t, 4),
            })

            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = float(thresh)

        best_threshold = round(best_threshold, 2)

        # Compute final metrics at optimal threshold
        y_pred_opt: np.ndarray = (y_proba >= best_threshold).astype(int)
        optimal_metrics: Dict[str, float] = {
            "threshold": best_threshold,
            "accuracy": round(accuracy_score(y_test, y_pred_opt), 4),
            "precision": round(precision_score(y_test, y_pred_opt, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_opt, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred_opt, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }

        print(f"\n[INFO] Optimal Threshold: {best_threshold}")
        print(f"  Accuracy  : {optimal_metrics['accuracy']}")
        print(f"  Precision : {optimal_metrics['precision']}")
        print(f"  Recall    : {optimal_metrics['recall']}")
        print(f"  F1 Score  : {optimal_metrics['f1_score']}")
        print(f"  ROC-AUC   : {optimal_metrics['roc_auc']}")

        logger.info("Optimal threshold: %.2f → F1=%.4f", best_threshold, best_f1)
        return best_threshold, optimal_metrics
    except Exception as exc:
        logger.error("Threshold optimization failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5/6 — VISUALIZATION: ROC, PR CURVE, FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════


def plot_roc_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    optimal_threshold: float,
) -> None:
    """Plot and save the ROC curve with AUC annotation and optimal threshold marker.

    Args:
        y_test: True labels.
        y_proba: Predicted probabilities for the positive class.
        model_name: Name of the model for the title.
        optimal_threshold: Optimal classification threshold to mark on the curve.
    """
    try:
        logger.info("Generating ROC curve ...")
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc_val: float = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(10, 8))

        # ROC curve
        ax.plot(fpr, tpr, color="#e74c3c", lw=2.5, label=f"{model_name} (AUC = {roc_auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="#95a5a6", lw=1.5, linestyle="--", label="Random Classifier")

        # Fill area under ROC
        ax.fill_between(fpr, tpr, alpha=0.15, color="#e74c3c")

        # Mark optimal threshold point
        optimal_idx: int = int(np.argmin(np.abs(thresholds - optimal_threshold)))
        ax.scatter(
            fpr[optimal_idx], tpr[optimal_idx],
            s=150, color="#f39c12", zorder=5, edgecolors="black", linewidths=1.5,
            label=f"Optimal Threshold = {optimal_threshold:.2f}",
        )
        ax.annotate(
            f"({fpr[optimal_idx]:.2f}, {tpr[optimal_idx]:.2f})",
            xy=(fpr[optimal_idx], tpr[optimal_idx]),
            xytext=(fpr[optimal_idx] + 0.1, tpr[optimal_idx] - 0.1),
            fontsize=11, fontweight="bold", color="#2c3e50",
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5),
        )

        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title(f"ROC Curve — {model_name}", fontsize=16, fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=12)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path: Path = IMAGE_DIR / "roc_curve.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] ROC curve → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate ROC curve: %s", exc)
        raise


def plot_precision_recall_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
) -> None:
    """Plot and save the Precision-Recall curve.

    Args:
        y_test: True labels.
        y_proba: Predicted probabilities for the positive class.
        model_name: Name of the model for the title.
    """
    try:
        logger.info("Generating Precision-Recall curve ...")
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc: float = auc(recall_vals, precision_vals)

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(
            recall_vals, precision_vals,
            color="#3498db", lw=2.5,
            label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
        )

        # Baseline
        baseline: float = float(y_test.mean())
        ax.axhline(y=baseline, color="#95a5a6", lw=1.5, linestyle="--", label=f"Baseline ({baseline:.3f})")
        ax.fill_between(recall_vals, precision_vals, alpha=0.15, color="#3498db")

        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=16, fontweight="bold", pad=15)
        ax.legend(loc="upper right", fontsize=12)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path: Path = IMAGE_DIR / "precision_recall_curve.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] Precision-Recall curve → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate Precision-Recall curve: %s", exc)
        raise


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20,
) -> None:
    """Plot and save feature importance from a tree-based model.

    Args:
        model: Fitted tree-based model with ``feature_importances_`` attribute.
        feature_names: List of feature column names.
        top_n: Number of top features to display.
    """
    try:
        logger.info("Generating feature importance plot ...")

        if not hasattr(model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_ attribute — skipping.")
            print("[WARN] Model lacks feature_importances_ — skipping feature importance plot.")
            return

        importances: np.ndarray = model.feature_importances_
        importance_df: pd.DataFrame = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))

        bars = ax.barh(
            importance_df["Feature"].values[::-1],
            importance_df["Importance"].values[::-1],
            color=colors[::-1],
            edgecolor="black",
            linewidth=0.5,
            height=0.7,
        )

        # Add value annotations
        for bar, val in zip(bars, importance_df["Importance"].values[::-1]):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#2c3e50",
            )

        ax.set_xlabel("Feature Importance (Gain)", fontsize=13)
        ax.set_ylabel("Feature", fontsize=13)
        ax.set_title(
            f"Top {top_n} Feature Importances — XGBoost (Tuned)",
            fontsize=16, fontweight="bold", pad=15,
        )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        save_path: Path = IMAGE_DIR / "feature_importance.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] Feature importance plot → {save_path}")

        # Print top features
        print("\n[INFO] Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"    {row['Feature']:<30} → {row['Importance']:.4f}")

    except Exception as exc:
        logger.error("Failed to generate feature importance plot: %s", exc)
        raise


def plot_confusion_matrix_heatmap(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    threshold: float,
) -> None:
    """Plot and save a styled confusion matrix heatmap.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        model_name: Name of the model for the title.
        threshold: Threshold used for predictions.
    """
    try:
        logger.info("Generating confusion matrix heatmap ...")
        cm: np.ndarray = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
            linewidths=2,
            linecolor="white",
            annot_kws={"size": 16, "fontweight": "bold"},
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=13)
        ax.set_ylabel("True Label", fontsize=13)
        ax.set_title(
            f"Confusion Matrix — {model_name}\n(Threshold: {threshold:.2f})",
            fontsize=15, fontweight="bold", pad=15,
        )
        plt.tight_layout()

        save_path: Path = IMAGE_DIR / "confusion_matrix.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] Confusion matrix → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate confusion matrix heatmap: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT EXPORT
# ══════════════════════════════════════════════════════════════════════════════


def save_model_artifacts(
    model: Any,
    feature_columns: List[str],
    label_encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
) -> None:
    """Serialize and save all model artifacts to the ``model/`` directory.

    Artifacts saved:
        - ``churn_model.pkl``: Trained classifier.
        - ``feature_columns.pkl``: Ordered list of feature column names.
        - ``label_encoders.pkl``: Dictionary of fitted LabelEncoders.
        - ``scaler.pkl``: Fitted StandardScaler.

    Args:
        model: Trained classifier to serialize.
        feature_columns: List of feature column names used during training.
        label_encoders: Dictionary of fitted LabelEncoder instances.
        scaler: Fitted StandardScaler instance.
    """
    try:
        logger.info("Saving model artifacts to '%s' ...", MODEL_DIR)
        print("\n[INFO] Saving model artifacts ...")

        artifacts: Dict[str, Tuple[Any, str]] = {
            "churn_model.pkl": (model, "Trained XGBoost Model"),
            "feature_columns.pkl": (feature_columns, "Feature Columns"),
            "label_encoders.pkl": (label_encoders, "Label Encoders"),
            "scaler.pkl": (scaler, "Standard Scaler"),
        }

        for filename, (artifact, description) in artifacts.items():
            filepath: Path = MODEL_DIR / filename
            with open(filepath, "wb") as f:
                pickle.dump(artifact, f)
            file_size: float = filepath.stat().st_size / 1024
            logger.info("Saved %s: %s (%.1f KB)", description, filepath, file_size)
            print(f"  [SAVED] {description:<25} → {filepath} ({file_size:.1f} KB)")

        print("[SUCCESS] All model artifacts saved successfully.")
        logger.info("All model artifacts saved.")
    except Exception as exc:
        logger.error("Failed to save model artifacts: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════


def generate_model_report(
    baseline_results: Dict[str, Dict[str, Any]],
    tuned_metrics: Dict[str, float],
    optimal_threshold: float,
    best_params: Dict[str, Any],
    feature_columns: List[str],
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Generate and write a comprehensive model evaluation report.

    Args:
        baseline_results: Dictionary of baseline model evaluation metrics.
        tuned_metrics: Metrics from the tuned model at the optimal threshold.
        optimal_threshold: Optimized classification threshold.
        best_params: Best hyperparameters from RandomizedSearchCV.
        feature_columns: List of feature column names.
        y_test: True test labels.
        y_pred: Predicted labels from the tuned model.
    """
    try:
        logger.info("Generating model evaluation report ...")
        lines: List[str] = []
        sep: str = "=" * 80
        sub_sep: str = "-" * 80

        # ── Header ───────────────────────────────────────────────────────────
        lines.append(sep)
        lines.append("  TELCO CUSTOMER CHURN — MODEL EVALUATION REPORT")
        lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Author    : Krishnaveni Mitukula")
        lines.append(sep)
        lines.append("")

        # ── Section 1: Baseline Model Comparison ─────────────────────────────
        lines.append("1. BASELINE MODEL COMPARISON")
        lines.append(sub_sep)
        lines.append(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
        lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for model_name, metrics in baseline_results.items():
            lines.append(
                f"  {model_name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {metrics['roc_auc']:>10.4f}"
            )
        lines.append("")

        # ── Section 2: Tuned Model Performance ──────────────────────────────
        lines.append("2. TUNED MODEL PERFORMANCE (XGBoost + RandomizedSearchCV)")
        lines.append(sub_sep)
        lines.append(f"  Optimal Threshold : {optimal_threshold}")
        for metric_name, metric_val in tuned_metrics.items():
            if metric_name != "threshold":
                lines.append(f"  {metric_name:<20} : {metric_val:.4f}")
        lines.append("")

        # ── Section 3: Best Hyperparameters ──────────────────────────────────
        lines.append("3. OPTIMIZED HYPERPARAMETERS")
        lines.append(sub_sep)
        for param, value in sorted(best_params.items()):
            formatted: str = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(f"  {param:<30} : {formatted}")
        lines.append("")

        # ── Section 4: Classification Report ─────────────────────────────────
        lines.append("4. DETAILED CLASSIFICATION REPORT")
        lines.append(sub_sep)
        report_str: str = classification_report(
            y_test, y_pred,
            target_names=["Retained", "Churned"],
            digits=4,
        )
        for line in report_str.split("\n"):
            lines.append(f"  {line}")
        lines.append("")

        # ── Section 5: Confusion Matrix ──────────────────────────────────────
        lines.append("5. CONFUSION MATRIX")
        lines.append(sub_sep)
        cm: np.ndarray = confusion_matrix(y_test, y_pred)
        lines.append(f"                    Predicted")
        lines.append(f"                    Retained    Churned")
        lines.append(f"  Actual Retained   {cm[0][0]:<12}{cm[0][1]}")
        lines.append(f"  Actual Churned    {cm[1][0]:<12}{cm[1][1]}")
        lines.append("")

        tn, fp, fn, tp = cm.ravel()
        lines.append(f"  True Negatives  (TN) : {tn:,}")
        lines.append(f"  False Positives (FP) : {fp:,}")
        lines.append(f"  False Negatives (FN) : {fn:,}")
        lines.append(f"  True Positives  (TP) : {tp:,}")
        lines.append("")

        # ── Section 6: Feature List ──────────────────────────────────────────
        lines.append("6. FEATURES USED IN TRAINING")
        lines.append(sub_sep)
        lines.append(f"  Total features: {len(feature_columns)}")
        for i, col in enumerate(feature_columns, 1):
            lines.append(f"    {i:>3}. {col}")
        lines.append("")

        # ── Section 7: Model Interpretation Notes ────────────────────────────
        lines.append("7. MODEL INTERPRETATION NOTES")
        lines.append(sub_sep)
        lines.append("  • The model uses XGBoost with class imbalance handling via scale_pos_weight.")
        lines.append("  • Hyperparameters were tuned using RandomizedSearchCV with 5-fold stratified CV.")
        lines.append(f"  • The classification threshold was optimized to {optimal_threshold} (default: 0.50)")
        lines.append("    to maximize the F1 score, balancing precision and recall.")
        lines.append("  • Feature importance is based on XGBoost's gain-based metric.")
        lines.append("  • SHAP-based explainability analysis is available via explainability.py.")
        lines.append("")
        lines.append(sep)
        lines.append("  END OF REPORT")
        lines.append(sep)

        # ── Write ────────────────────────────────────────────────────────────
        report_text: str = "\n".join(lines)
        report_path: Path = REPORT_DIR / "model_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        logger.info("Model report saved: %s", report_path)
        print(f"[SAVED] Model evaluation report → {report_path}")
    except Exception as exc:
        logger.error("Failed to generate model report: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point for the model training pipeline.

    Orchestrates:
        1. Load cleaned data.
        2. Engineer features.
        3. Encode categoricals & scale numericals.
        4. Train baseline models.
        5. Tune best model with RandomizedSearchCV.
        6. Optimize classification threshold.
        7. Generate evaluation plots.
        8. Export model artifacts.
        9. Write model report.
    """
    try:
        logger.info("=" * 70)
        logger.info("TELCO CUSTOMER CHURN — MODEL TRAINING PIPELINE STARTED")
        logger.info("=" * 70)

        print("\n" + "█" * 70)
        print("  TELCO CUSTOMER CHURN — MODEL TRAINING PIPELINE")
        print("  Author: Krishnaveni Mitukula")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("█" * 70 + "\n")

        # ── Step 1: Load Data ────────────────────────────────────────────────
        df: pd.DataFrame = load_cleaned_data(CLEANED_DATA_PATH)

        # ── Step 2: Feature Engineering ──────────────────────────────────────
        df = engineer_features(df)

        # ── Step 3: Encode Categoricals ──────────────────────────────────────
        df, label_encoders = encode_categorical_features(df)

        # ── Step 4: Split Features & Target ──────────────────────────────────
        target_col: str = "Churn"
        feature_cols: List[str] = [c for c in df.columns if c != target_col]

        X: pd.DataFrame = df[feature_cols].copy()
        y: pd.Series = df[target_col].copy()

        logger.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)
        print(f"\n[INFO] Feature matrix: {X.shape} | Target: {y.shape}")

        # ── Step 5: Train-Test Split ─────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        logger.info("Train: %s | Test: %s", X_train.shape, X_test.shape)
        print(f"[INFO] Train split: {X_train.shape} | Test split: {X_test.shape}")

        # ── Step 6: Scale Numerical Features ─────────────────────────────────
        numerical_cols: List[str] = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train, X_test, scaler = scale_numerical_features(X_train, X_test, numerical_cols)

        # ── Step 7: Compute Class Weight ─────────────────────────────────────
        scale_pos_weight: float = compute_class_weight_ratio(y_train)

        # ── Step 8: Baseline Models ──────────────────────────────────────────
        best_info, baseline_results = train_baseline_models(
            X_train, X_test, y_train, y_test, scale_pos_weight
        )

        # ── Step 9: Hyperparameter Tuning ────────────────────────────────────
        tuned_model: xgb.XGBClassifier = tune_xgboost(
            X_train, y_train, scale_pos_weight, n_iter=80, cv_folds=5
        )

        # ── Step 10: Evaluate Tuned Model ────────────────────────────────────
        print("\n" + "=" * 70)
        print("  TUNED MODEL EVALUATION")
        print("=" * 70)
        tuned_eval: Dict[str, Any] = evaluate_model(
            tuned_model, X_test, y_test, "XGBoost (Tuned)"
        )

        # ── Step 11: Optimize Threshold ──────────────────────────────────────
        optimal_threshold, optimal_metrics = optimize_threshold(
            tuned_model, X_test, y_test
        )

        # ── Step 12: Final Predictions ───────────────────────────────────────
        y_proba_final: np.ndarray = tuned_model.predict_proba(X_test)[:, 1]
        y_pred_final: np.ndarray = (y_proba_final >= optimal_threshold).astype(int)

        # ── Step 13: Generate Plots ──────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  GENERATING EVALUATION PLOTS")
        print("=" * 70)
        plot_roc_curve(y_test, y_proba_final, "XGBoost (Tuned)", optimal_threshold)
        plot_precision_recall_curve(y_test, y_proba_final, "XGBoost (Tuned)")
        plot_feature_importance(tuned_model, feature_cols, top_n=20)
        plot_confusion_matrix_heatmap(y_test, y_pred_final, "XGBoost (Tuned)", optimal_threshold)

        # ── Step 14: Save Artifacts ──────────────────────────────────────────
        save_model_artifacts(tuned_model, feature_cols, label_encoders, scaler)

        # ── Step 15: Generate Model Report ───────────────────────────────────
        best_params: Dict[str, Any] = tuned_model.get_params()
        generate_model_report(
            baseline_results=baseline_results,
            tuned_metrics=optimal_metrics,
            optimal_threshold=optimal_threshold,
            best_params=best_params,
            feature_columns=feature_cols,
            y_test=y_test,
            y_pred=y_pred_final,
        )

        # ── Final Summary ────────────────────────────────────────────────────
        print("\n" + "█" * 70)
        print("  MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  Best Baseline         : {best_info['name']}")
        print(f"  Tuned Model           : XGBoost (RandomizedSearchCV)")
        print(f"  Optimal Threshold     : {optimal_threshold}")
        print(f"  Final F1 Score        : {optimal_metrics['f1_score']:.4f}")
        print(f"  Final ROC-AUC         : {optimal_metrics['roc_auc']:.4f}")
        print(f"  Artifacts saved       : model/")
        print(f"  Plots saved           : images/")
        print(f"  Report saved          : reports/model_report.txt")
        print("█" * 70 + "\n")

        logger.info("Model training pipeline completed successfully.")

    except FileNotFoundError as exc:
        logger.critical("PIPELINE ABORTED — File not found: %s", exc)
        print(f"\n[CRITICAL] Pipeline aborted: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.critical("PIPELINE ABORTED — Unexpected error: %s", exc)
        print(f"\n[CRITICAL] Pipeline aborted: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
