"""
eda_analysis.py
================================================================================
Telco Customer Churn — Exploratory Data Analysis & Data Cleaning Pipeline
================================================================================

Author  : Krishnaveni Mitukula
Version : 1.0.0
Python  : 3.11+

Description
-----------
Production-grade EDA module that:
  1. Loads and profiles the Telco Customer Churn dataset.
  2. Cleans & imputes problematic columns (TotalCharges).
  3. Encodes the target variable.
  4. Generates publication-quality visualizations saved to ``images/``.
  5. Writes a structured business-insights report to ``reports/``.

Usage
-----
    python eda_analysis.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import os
import sys
import textwrap
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns

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
IMAGE_DIR: Path = BASE_DIR / "images"
REPORT_DIR: Path = BASE_DIR / "reports"
DATASET_PATH: Path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Ensure output directories exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# VISUAL STYLE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    palette="muted",
    font_scale=1.15,
    rc={
        "figure.figsize": (14, 7),
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    },
)

CHURN_PALETTE: Dict[int, str] = {0: "#2ecc71", 1: "#e74c3c"}
CHURN_LABELS: Dict[int, str] = {0: "Retained", 1: "Churned"}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOADING & PROFILING
# ══════════════════════════════════════════════════════════════════════════════


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load the Telco Customer Churn CSV dataset from disk.

    Args:
        filepath: Absolute or relative ``Path`` to the CSV file.

    Returns:
        Raw ``pd.DataFrame`` loaded from the CSV.

    Raises:
        FileNotFoundError: If the dataset file does not exist at *filepath*.
        pd.errors.ParserError: If the CSV cannot be parsed.
    """
    try:
        logger.info("Loading dataset from: %s", filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. "
                "Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' inside the 'data/' folder."
            )
        df: pd.DataFrame = pd.read_csv(filepath)
        logger.info(
            "Dataset loaded successfully — Shape: %s | Columns: %d",
            df.shape,
            len(df.columns),
        )
        print(f"[INFO] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError as exc:
        logger.error("FileNotFoundError: %s", exc)
        raise
    except pd.errors.ParserError as exc:
        logger.error("CSV parsing failed: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error during dataset loading: %s", exc)
        raise


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive profile of the dataset.

    Args:
        df: Input ``DataFrame`` to profile.

    Returns:
        Dictionary containing profiling metrics: shape, dtypes, null counts,
        duplicate count, memory usage, and descriptive statistics.
    """
    try:
        logger.info("Profiling dataset ...")
        null_profile: pd.Series = df.isnull().sum()
        null_pct: pd.Series = (df.isnull().sum() / len(df) * 100).round(2)

        profile: Dict[str, Any] = {
            "shape": df.shape,
            "dtypes": df.dtypes.value_counts().to_dict(),
            "null_counts": null_profile[null_profile > 0].to_dict(),
            "null_pct": null_pct[null_pct > 0].to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

        print(f"[INFO] Shape           : {profile['shape']}")
        print(f"[INFO] Column dtypes   : {profile['dtypes']}")
        print(f"[INFO] Null columns    : {profile['null_counts'] if profile['null_counts'] else 'None'}")
        print(f"[INFO] Duplicate rows  : {profile['duplicates']}")
        print(f"[INFO] Memory usage    : {profile['memory_mb']} MB")

        logger.info("Profiling complete.")
        return profile
    except Exception as exc:
        logger.error("Error during dataset profiling: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATA CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the 'TotalCharges' column.

    Steps:
        1. Strip leading/trailing whitespace.
        2. Coerce non-numeric values to ``NaN``.
        3. Impute missing values using the **median TotalCharges per tenure cohort**.

    Args:
        df: DataFrame containing raw ``TotalCharges`` column.

    Returns:
        DataFrame with cleaned ``TotalCharges`` as ``float64``.
    """
    try:
        logger.info("Cleaning 'TotalCharges' column ...")
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        null_count_before: int = int(df["TotalCharges"].isnull().sum())
        logger.info("Null values in TotalCharges after coercion: %d", null_count_before)
        print(f"[INFO] TotalCharges — NaN values detected: {null_count_before}")

        if null_count_before > 0:
            # Create tenure cohorts for targeted imputation
            tenure_bins: List[int] = [0, 12, 24, 48, 60, 73]
            tenure_labels: List[str] = ["0-12", "13-24", "25-48", "49-60", "60+"]
            df["_tenure_cohort"] = pd.cut(
                df["tenure"], bins=tenure_bins, labels=tenure_labels, right=True
            )

            # Compute median TotalCharges per cohort
            cohort_medians: pd.Series = df.groupby("_tenure_cohort", observed=False)[
                "TotalCharges"
            ].median()
            logger.info("Tenure-cohort medians for imputation:\n%s", cohort_medians.to_string())
            print("[INFO] Cohort medians for TotalCharges imputation:")
            for cohort, median_val in cohort_medians.items():
                print(f"       Tenure {cohort} months → Median = ${median_val:,.2f}")

            # Impute nulls using cohort median
            for cohort_label in tenure_labels:
                mask = (df["_tenure_cohort"] == cohort_label) & (df["TotalCharges"].isnull())
                median_value: float = cohort_medians.get(cohort_label, df["TotalCharges"].median())
                if pd.isna(median_value):
                    median_value = df["TotalCharges"].median()
                df.loc[mask, "TotalCharges"] = median_value
                filled: int = int(mask.sum())
                if filled > 0:
                    logger.info(
                        "Imputed %d rows in cohort '%s' with median $%.2f",
                        filled,
                        cohort_label,
                        median_value,
                    )

            # Drop helper column
            df.drop(columns=["_tenure_cohort"], inplace=True)

            # Global median fallback for any remaining NaNs
            remaining_nulls: int = int(df["TotalCharges"].isnull().sum())
            if remaining_nulls > 0:
                global_median: float = df["TotalCharges"].median()
                df["TotalCharges"] = df["TotalCharges"].fillna(global_median)
                logger.info(
                    "Applied global median fallback ($%.2f) for %d remaining NaN values.",
                    global_median, remaining_nulls,
                )
                print(f"[INFO] Global median fallback applied for {remaining_nulls} remaining NaN values.")

        null_count_after: int = int(df["TotalCharges"].isnull().sum())
        logger.info("TotalCharges cleaning complete — Remaining NaN: %d", null_count_after)
        print(f"[INFO] TotalCharges — Remaining NaN after imputation: {null_count_after}")
        return df
    except KeyError as exc:
        logger.error("Column not found: %s", exc)
        raise
    except Exception as exc:
        logger.error("Error cleaning TotalCharges: %s", exc)
        raise


def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 'customerID' column as it carries no predictive value.

    Args:
        df: DataFrame potentially containing ``customerID``.

    Returns:
        DataFrame without the ``customerID`` column.
    """
    try:
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
            logger.info("Dropped 'customerID' column.")
            print("[INFO] Dropped 'customerID' column (non-predictive identifier).")
        else:
            logger.warning("'customerID' column not found — skipping drop.")
            print("[WARN] 'customerID' column not found — skipping.")
        return df
    except Exception as exc:
        logger.error("Error dropping customerID: %s", exc)
        raise


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the 'Churn' target column: 'Yes' → 1, 'No' → 0.

    Args:
        df: DataFrame with raw ``Churn`` column ('Yes'/'No' strings).

    Returns:
        DataFrame with ``Churn`` column encoded as ``int`` (0/1).
    """
    try:
        logger.info("Encoding target variable 'Churn' ...")
        churn_mapping: Dict[str, int] = {"Yes": 1, "No": 0}
        df["Churn"] = df["Churn"].map(churn_mapping)

        unmapped: int = int(df["Churn"].isnull().sum())
        if unmapped > 0:
            logger.warning("%d rows with unmapped Churn values detected.", unmapped)
            df["Churn"] = df["Churn"].fillna(0).astype(int)
        else:
            df["Churn"] = df["Churn"].astype(int)

        churn_dist: pd.Series = df["Churn"].value_counts()
        churn_pct: pd.Series = df["Churn"].value_counts(normalize=True).mul(100).round(2)
        print("[INFO] Churn target encoded successfully:")
        print(f"       Retained (0): {churn_dist.get(0, 0):,} ({churn_pct.get(0, 0):.1f}%)")
        print(f"       Churned  (1): {churn_dist.get(1, 0):,} ({churn_pct.get(1, 0):.1f}%)")
        logger.info("Target encoding complete. Distribution: %s", churn_dist.to_dict())
        return df
    except KeyError as exc:
        logger.error("'Churn' column not found: %s", exc)
        raise
    except Exception as exc:
        logger.error("Error encoding target variable: %s", exc)
        raise


def run_data_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full data cleaning pipeline sequentially.

    Pipeline:
        1. Clean & impute ``TotalCharges``.
        2. Drop ``customerID``.
        3. Encode ``Churn`` target.

    Args:
        df: Raw loaded DataFrame.

    Returns:
        Cleaned DataFrame ready for EDA visualization.
    """
    try:
        logger.info("=" * 70)
        logger.info("STARTING DATA CLEANING PIPELINE")
        logger.info("=" * 70)
        print("\n" + "=" * 70)
        print("  DATA CLEANING PIPELINE")
        print("=" * 70)

        df = clean_total_charges(df)
        df = drop_customer_id(df)
        df = encode_target(df)

        logger.info("Data cleaning pipeline completed successfully.")
        print(f"\n[SUCCESS] Cleaning complete — Final shape: {df.shape}")
        print("=" * 70 + "\n")
        return df
    except Exception as exc:
        logger.error("Data cleaning pipeline failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Generate a side-by-side Pie Chart + Count Plot of churn distribution.

    Saves the figure to ``images/churn_distribution.png``.

    Args:
        df: Cleaned DataFrame with encoded ``Churn`` column.
    """
    try:
        logger.info("Generating churn distribution plot ...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # ── Pie Chart ────────────────────────────────────────────────────────
        churn_counts: pd.Series = df["Churn"].value_counts()
        labels: List[str] = [CHURN_LABELS[idx] for idx in churn_counts.index]
        colors: List[str] = [CHURN_PALETTE[idx] for idx in churn_counts.index]
        explode: Tuple[float, ...] = (0.04, 0.04)

        wedges, texts, autotexts = axes[0].pie(
            churn_counts.values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            explode=explode,
            shadow=True,
            textprops={"fontsize": 13, "fontweight": "bold"},
        )
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_color("white")
            autotext.set_fontweight("bold")
        axes[0].set_title("Churn Distribution (Pie Chart)", fontsize=15, fontweight="bold", pad=15)

        # ── Count Plot ───────────────────────────────────────────────────────
        churn_labels_mapped: pd.Series = df["Churn"].map(CHURN_LABELS)
        sns.countplot(
            x=churn_labels_mapped,
            palette=list(CHURN_PALETTE.values()),
            ax=axes[1],
            edgecolor="black",
            linewidth=1.2,
            order=["Retained", "Churned"],
        )
        axes[1].set_title("Churn Distribution (Count Plot)", fontsize=15, fontweight="bold", pad=15)
        axes[1].set_xlabel("Customer Status", fontsize=13)
        axes[1].set_ylabel("Count", fontsize=13)

        # Add value annotations on bars
        for patch in axes[1].patches:
            height: float = patch.get_height()
            axes[1].annotate(
                f"{int(height):,}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="#2c3e50",
            )

        plt.suptitle(
            "Customer Churn Overview",
            fontsize=18,
            fontweight="bold",
            y=1.02,
            color="#2c3e50",
        )
        plt.tight_layout()

        save_path: Path = IMAGE_DIR / "churn_distribution.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] Churn distribution plot → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate churn distribution plot: %s", exc)
        raise


def plot_categorical_drivers(df: pd.DataFrame) -> None:
    """Generate segmented count plots for key categorical features vs Churn.

    Features plotted: Gender, Contract, InternetService, PaymentMethod.
    Saves the figure to ``images/categorical_drivers.png``.

    Args:
        df: Cleaned DataFrame with encoded ``Churn`` column.
    """
    try:
        logger.info("Generating categorical driver plots ...")
        categorical_features: List[str] = [
            "gender",
            "Contract",
            "InternetService",
            "PaymentMethod",
        ]

        # Validate columns exist
        missing_cols: List[str] = [c for c in categorical_features if c not in df.columns]
        if missing_cols:
            logger.warning("Missing categorical columns: %s — skipping them.", missing_cols)
            categorical_features = [c for c in categorical_features if c in df.columns]

        if not categorical_features:
            logger.warning("No categorical features available for plotting.")
            return

        n_features: int = len(categorical_features)
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes_flat: np.ndarray = axes.flatten()

        churn_label_col: str = "_churn_label"
        df[churn_label_col] = df["Churn"].map(CHURN_LABELS)

        for idx, feature in enumerate(categorical_features):
            ax = axes_flat[idx]
            sns.countplot(
                data=df,
                x=feature,
                hue=churn_label_col,
                palette=list(CHURN_PALETTE.values()),
                ax=ax,
                edgecolor="black",
                linewidth=0.8,
                hue_order=["Retained", "Churned"],
            )
            ax.set_title(f"{feature} vs Churn", fontsize=14, fontweight="bold", pad=10)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.legend(title="Status", fontsize=10, title_fontsize=11)

            # Rotate x-tick labels for long category names
            if feature in ("PaymentMethod", "InternetService"):
                ax.tick_params(axis="x", rotation=25)

            # Add value annotations on bars
            for patch in ax.patches:
                height = patch.get_height()
                if height > 0:
                    ax.annotate(
                        f"{int(height):,}",
                        (patch.get_x() + patch.get_width() / 2.0, height),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="#34495e",
                    )

        # Hide unused subplots if fewer than 4 features
        for idx in range(n_features, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Clean up temp column
        df.drop(columns=[churn_label_col], inplace=True)

        plt.suptitle(
            "Categorical Feature Analysis — Churn Segmentation",
            fontsize=18,
            fontweight="bold",
            y=1.01,
            color="#2c3e50",
        )
        plt.tight_layout()

        save_path: Path = IMAGE_DIR / "categorical_drivers.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] Categorical drivers plot → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate categorical driver plots: %s", exc)
        raise


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Generate a correlation heatmap for all numerical variables.

    Saves the figure to ``images/heatmap.png``.

    Args:
        df: Cleaned DataFrame.
    """
    try:
        logger.info("Generating correlation heatmap ...")
        numerical_cols: List[str] = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) < 2:
            logger.warning("Fewer than 2 numerical columns — skipping heatmap.")
            print("[WARN] Skipping heatmap — insufficient numerical columns.")
            return

        corr_matrix: pd.DataFrame = df[numerical_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 10))

        mask: np.ndarray = np.triu(np.ones_like(corr_matrix, dtype=bool))

        cmap = sns.diverging_palette(250, 15, s=75, l=40, n=12, center="light", as_cmap=True)

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
            ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title(
            "Correlation Heatmap — Numerical Features",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        save_path: Path = IMAGE_DIR / "heatmap.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] Correlation heatmap → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate correlation heatmap: %s", exc)
        raise


def plot_tenure_boxplot(df: pd.DataFrame) -> None:
    """Generate a boxplot comparing Tenure distributions across Churn status.

    Saves the figure to ``images/tenure_boxplot.png``.

    Args:
        df: Cleaned DataFrame with ``tenure`` and ``Churn`` columns.
    """
    try:
        logger.info("Generating tenure boxplot ...")
        fig, ax = plt.subplots(figsize=(10, 7))

        churn_labels_mapped: pd.Series = df["Churn"].map(CHURN_LABELS)

        sns.boxplot(
            x=churn_labels_mapped,
            y=df["tenure"],
            palette=list(CHURN_PALETTE.values()),
            width=0.5,
            linewidth=1.5,
            fliersize=4,
            ax=ax,
            order=["Retained", "Churned"],
        )
        sns.stripplot(
            x=churn_labels_mapped,
            y=df["tenure"],
            color="#2c3e50",
            alpha=0.15,
            size=3,
            jitter=True,
            ax=ax,
            order=["Retained", "Churned"],
        )
        ax.set_title(
            "Customer Tenure Distribution by Churn Status",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Customer Status", fontsize=13)
        ax.set_ylabel("Tenure (Months)", fontsize=13)

        # Add median annotations
        medians: pd.Series = df.groupby("Churn")["tenure"].median()
        for churn_val, x_pos in zip([0, 1], [0, 1]):
            median_val: float = medians.get(churn_val, 0)
            ax.annotate(
                f"Median: {median_val:.0f} mo",
                xy=(x_pos, median_val),
                xytext=(x_pos + 0.3, median_val + 5),
                fontsize=11,
                fontweight="bold",
                color=CHURN_PALETTE[churn_val],
                arrowprops=dict(arrowstyle="->", color=CHURN_PALETTE[churn_val], lw=1.5),
            )

        plt.tight_layout()
        save_path: Path = IMAGE_DIR / "tenure_boxplot.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] Tenure boxplot → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate tenure boxplot: %s", exc)
        raise


def plot_monthlycharges_boxplot(df: pd.DataFrame) -> None:
    """Generate a boxplot comparing MonthlyCharges across Churn status.

    Saves the figure to ``images/monthlycharges_boxplot.png``.

    Args:
        df: Cleaned DataFrame with ``MonthlyCharges`` and ``Churn`` columns.
    """
    try:
        logger.info("Generating MonthlyCharges boxplot ...")
        fig, ax = plt.subplots(figsize=(10, 7))

        churn_labels_mapped: pd.Series = df["Churn"].map(CHURN_LABELS)

        sns.boxplot(
            x=churn_labels_mapped,
            y=df["MonthlyCharges"],
            palette=list(CHURN_PALETTE.values()),
            width=0.5,
            linewidth=1.5,
            fliersize=4,
            ax=ax,
            order=["Retained", "Churned"],
        )
        sns.stripplot(
            x=churn_labels_mapped,
            y=df["MonthlyCharges"],
            color="#2c3e50",
            alpha=0.15,
            size=3,
            jitter=True,
            ax=ax,
            order=["Retained", "Churned"],
        )
        ax.set_title(
            "Monthly Charges Distribution by Churn Status",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Customer Status", fontsize=13)
        ax.set_ylabel("Monthly Charges ($)", fontsize=13)

        # Add median annotations
        medians: pd.Series = df.groupby("Churn")["MonthlyCharges"].median()
        for churn_val, x_pos in zip([0, 1], [0, 1]):
            median_val: float = medians.get(churn_val, 0)
            ax.annotate(
                f"Median: ${median_val:.2f}",
                xy=(x_pos, median_val),
                xytext=(x_pos + 0.3, median_val + 8),
                fontsize=11,
                fontweight="bold",
                color=CHURN_PALETTE[churn_val],
                arrowprops=dict(arrowstyle="->", color=CHURN_PALETTE[churn_val], lw=1.5),
            )

        plt.tight_layout()
        save_path: Path = IMAGE_DIR / "monthlycharges_boxplot.png"
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", save_path)
        print(f"[SAVED] MonthlyCharges boxplot → {save_path}")
    except Exception as exc:
        logger.error("Failed to generate MonthlyCharges boxplot: %s", exc)
        raise


def plot_totalcharges_distribution(df: pd.DataFrame) -> None:
    """Generate an interactive Plotly distribution plot for TotalCharges by Churn.

    Saves the figure to ``images/totalcharges_distribution.html`` (interactive)
    and ``images/totalcharges_distribution.png`` (static).

    Args:
        df: Cleaned DataFrame with ``TotalCharges`` and ``Churn`` columns.
    """
    try:
        logger.info("Generating TotalCharges distribution plot (Plotly) ...")
        plot_df: pd.DataFrame = df.copy()
        plot_df["Churn Status"] = plot_df["Churn"].map(CHURN_LABELS)

        fig = px.histogram(
            plot_df,
            x="TotalCharges",
            color="Churn Status",
            marginal="box",
            nbins=60,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"},
            title="Total Charges Distribution by Churn Status",
            labels={"TotalCharges": "Total Charges ($)", "count": "Frequency"},
            category_orders={"Churn Status": ["Retained", "Churned"]},
        )
        fig.update_layout(
            template="plotly_white",
            title_font_size=20,
            title_font_color="#2c3e50",
            font=dict(size=13),
            legend_title_text="Customer Status",
            xaxis_title="Total Charges ($)",
            yaxis_title="Frequency",
            width=1100,
            height=600,
        )

        # Save static image
        static_path: Path = IMAGE_DIR / "totalcharges_distribution.png"
        try:
            fig.write_image(str(static_path), scale=2)
            logger.info("Saved static Plotly image: %s", static_path)
            print(f"[SAVED] TotalCharges distribution (static) → {static_path}")
        except Exception as img_exc:
            logger.warning(
                "Could not save static Plotly image (kaleido may not be installed): %s",
                img_exc,
            )
            print(f"[WARN] Static export skipped (install kaleido for PNG export): {img_exc}")

        # Save interactive HTML
        html_path: Path = IMAGE_DIR / "totalcharges_distribution.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        logger.info("Saved interactive Plotly HTML: %s", html_path)
        print(f"[SAVED] TotalCharges distribution (interactive) → {html_path}")

    except Exception as exc:
        logger.error("Failed to generate TotalCharges distribution: %s", exc)
        raise


def generate_all_visualizations(df: pd.DataFrame) -> None:
    """Orchestrate generation of all EDA visualizations.

    Args:
        df: Cleaned DataFrame.
    """
    try:
        logger.info("=" * 70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 70)
        print("\n" + "=" * 70)
        print("  VISUALIZATION GENERATION")
        print("=" * 70)

        plot_churn_distribution(df)
        plot_categorical_drivers(df)
        plot_correlation_heatmap(df)
        plot_tenure_boxplot(df)
        plot_monthlycharges_boxplot(df)
        plot_totalcharges_distribution(df)

        logger.info("All visualizations generated successfully.")
        print("\n[SUCCESS] All visualizations saved to 'images/' directory.")
        print("=" * 70 + "\n")
    except Exception as exc:
        logger.error("Visualization pipeline failed: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUSINESS INSIGHTS REPORT
# ══════════════════════════════════════════════════════════════════════════════


def generate_business_insights_report(df: pd.DataFrame) -> None:
    """Generate a structured business insights report and write to disk.

    The report covers:
        - Dataset overview and quality summary.
        - Churn distribution statistics.
        - Categorical feature analysis (churn rates per category).
        - Continuous feature analysis (medians, correlations).
        - Key business observations and strategic recommendations.

    Saves the report to ``reports/business_insights.txt``.

    Args:
        df: Cleaned DataFrame with encoded ``Churn`` column.
    """
    try:
        logger.info("Generating business insights report ...")
        report_lines: List[str] = []
        separator: str = "=" * 80
        sub_separator: str = "-" * 80

        # ── Header ───────────────────────────────────────────────────────────
        report_lines.append(separator)
        report_lines.append("  TELCO CUSTOMER CHURN — BUSINESS INSIGHTS REPORT")
        report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"  Author   : Krishnaveni Mitukula")
        report_lines.append(separator)
        report_lines.append("")

        # ── Section 1: Dataset Overview ──────────────────────────────────────
        report_lines.append("1. DATASET OVERVIEW")
        report_lines.append(sub_separator)
        report_lines.append(f"   Total Records         : {len(df):,}")
        report_lines.append(f"   Total Features        : {len(df.columns)}")
        report_lines.append(f"   Numerical Features    : {len(df.select_dtypes(include=[np.number]).columns)}")
        report_lines.append(f"   Categorical Features  : {len(df.select_dtypes(include=['object']).columns)}")
        report_lines.append(f"   Missing Values        : {int(df.isnull().sum().sum())}")
        report_lines.append(f"   Duplicate Rows        : {int(df.duplicated().sum())}")
        report_lines.append("")

        # ── Section 2: Churn Distribution ────────────────────────────────────
        churn_counts: pd.Series = df["Churn"].value_counts()
        churn_pct: pd.Series = df["Churn"].value_counts(normalize=True).mul(100).round(2)
        total_customers: int = len(df)

        report_lines.append("2. CHURN DISTRIBUTION")
        report_lines.append(sub_separator)
        report_lines.append(f"   Retained Customers    : {churn_counts.get(0, 0):,} ({churn_pct.get(0, 0):.1f}%)")
        report_lines.append(f"   Churned Customers     : {churn_counts.get(1, 0):,} ({churn_pct.get(1, 0):.1f}%)")
        report_lines.append(f"   Class Imbalance Ratio  : 1:{churn_counts.get(0, 1) / max(churn_counts.get(1, 1), 1):.2f} (Churned:Retained)")
        report_lines.append("")

        # ── Section 3: Categorical Feature Analysis ──────────────────────────
        report_lines.append("3. CATEGORICAL FEATURE ANALYSIS (Churn Rate by Category)")
        report_lines.append(sub_separator)

        categorical_features: List[str] = [
            "gender", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod",
        ]

        for feature in categorical_features:
            if feature not in df.columns:
                continue
            report_lines.append(f"\n   {feature}:")
            cross_tab: pd.DataFrame = pd.crosstab(
                df[feature], df["Churn"], normalize="index"
            ).mul(100).round(1)
            counts: pd.Series = df[feature].value_counts()
            for category in cross_tab.index:
                churn_rate: float = cross_tab.loc[category, 1] if 1 in cross_tab.columns else 0.0
                count: int = int(counts.get(category, 0))
                indicator: str = "⚠ HIGH" if churn_rate > 35 else "  "
                report_lines.append(
                    f"     {str(category):<35} → Churn Rate: {churn_rate:5.1f}%  (n={count:,}) {indicator}"
                )

        report_lines.append("")

        # ── Section 4: Continuous Feature Analysis ───────────────────────────
        report_lines.append("4. CONTINUOUS FEATURE ANALYSIS")
        report_lines.append(sub_separator)

        continuous_features: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
        for feature in continuous_features:
            if feature not in df.columns:
                continue
            retained_stats: pd.Series = df[df["Churn"] == 0][feature].describe()
            churned_stats: pd.Series = df[df["Churn"] == 1][feature].describe()
            report_lines.append(f"\n   {feature}:")
            report_lines.append(f"     Retained — Mean: {retained_stats['mean']:>10,.2f} | Median: {retained_stats['50%']:>10,.2f} | Std: {retained_stats['std']:>10,.2f}")
            report_lines.append(f"     Churned  — Mean: {churned_stats['mean']:>10,.2f} | Median: {churned_stats['50%']:>10,.2f} | Std: {churned_stats['std']:>10,.2f}")

        report_lines.append("")

        # ── Section 5: Correlation Insights ──────────────────────────────────
        report_lines.append("5. KEY CORRELATION INSIGHTS")
        report_lines.append(sub_separator)
        numerical_df: pd.DataFrame = df.select_dtypes(include=[np.number])
        if "Churn" in numerical_df.columns:
            churn_corr: pd.Series = (
                numerical_df.corr()["Churn"]
                .drop("Churn", errors="ignore")
                .sort_values(key=abs, ascending=False)
            )
            report_lines.append("   Top correlations with Churn:")
            for feat, corr_val in churn_corr.head(5).items():
                direction: str = "Positive" if corr_val > 0 else "Negative"
                report_lines.append(f"     {str(feat):<25} → r = {corr_val:+.4f} ({direction})")
        report_lines.append("")

        # ── Section 6: Key Business Observations ────────────────────────────
        report_lines.append("6. KEY BUSINESS OBSERVATIONS")
        report_lines.append(sub_separator)

        observations: List[str] = []

        # Contract analysis
        if "Contract" in df.columns:
            mtm_churn: float = df[df["Contract"] == "Month-to-month"]["Churn"].mean() * 100
            observations.append(
                f"   • Month-to-month contracts exhibit a {mtm_churn:.1f}% churn rate, "
                "significantly higher than annual or two-year contracts. "
                "This suggests contract lock-in is the strongest retention lever."
            )

        # Tenure analysis
        if "tenure" in df.columns:
            short_tenure_churn: float = df[df["tenure"] <= 12]["Churn"].mean() * 100
            long_tenure_churn: float = df[df["tenure"] > 48]["Churn"].mean() * 100
            observations.append(
                f"   • New customers (tenure ≤ 12 months) churn at {short_tenure_churn:.1f}%, "
                f"versus only {long_tenure_churn:.1f}% for loyal customers (tenure > 48 months). "
                "Early lifecycle engagement is critical."
            )

        # Internet service analysis
        if "InternetService" in df.columns:
            fiber_churn: float = df[df["InternetService"] == "Fiber optic"]["Churn"].mean() * 100
            dsl_churn: float = df[df["InternetService"] == "DSL"]["Churn"].mean() * 100
            observations.append(
                f"   • Fiber optic customers churn at {fiber_churn:.1f}% vs DSL at {dsl_churn:.1f}%. "
                "Higher expectations or pricing dissatisfaction may drive fiber optic attrition."
            )

        # Tech support analysis
        if "TechSupport" in df.columns:
            no_tech_churn: float = df[df["TechSupport"] == "No"]["Churn"].mean() * 100
            yes_tech_churn: float = df[df["TechSupport"] == "Yes"]["Churn"].mean() * 100
            observations.append(
                f"   • Customers without tech support churn at {no_tech_churn:.1f}% vs "
                f"{yes_tech_churn:.1f}% with support. Proactive tech support provisioning "
                "could meaningfully reduce attrition."
            )

        # Monthly charges analysis
        if "MonthlyCharges" in df.columns:
            high_charge_threshold: float = df["MonthlyCharges"].quantile(0.75)
            high_charge_churn: float = (
                df[df["MonthlyCharges"] >= high_charge_threshold]["Churn"].mean() * 100
            )
            observations.append(
                f"   • High-spend customers (MonthlyCharges ≥ ${high_charge_threshold:.2f}) "
                f"churn at {high_charge_churn:.1f}%. Value perception and competitive "
                "benchmarking should be prioritized for this segment."
            )

        # Payment method analysis
        if "PaymentMethod" in df.columns:
            echeck_churn: float = (
                df[df["PaymentMethod"] == "Electronic check"]["Churn"].mean() * 100
            )
            observations.append(
                f"   • Electronic check users churn at {echeck_churn:.1f}%, the highest "
                "among all payment methods. Incentivizing automatic payment enrollment "
                "may improve retention."
            )

        for obs in observations:
            report_lines.append(obs)
        report_lines.append("")

        # ── Section 7: Strategic Recommendations ────────────────────────────
        report_lines.append("7. STRATEGIC RECOMMENDATIONS")
        report_lines.append(sub_separator)
        recommendations: List[str] = [
            "   1. CONTRACT MIGRATION: Offer month-to-month customers incentives to",
            "      upgrade to annual or biennial contracts (e.g., 15-20% discount).",
            "",
            "   2. EARLY LIFECYCLE PROGRAM: Deploy targeted onboarding and engagement",
            "      campaigns during the first 12 months to reduce early churn.",
            "",
            "   3. TECH SUPPORT BUNDLES: Proactively offer tech support packages to",
            "      customers without current coverage, especially fiber optic users.",
            "",
            "   4. PAYMENT METHOD INCENTIVES: Encourage electronic check users to",
            "      switch to automatic credit card or bank transfer payments.",
            "",
            "   5. VALUE REINFORCEMENT: Implement loyalty rewards and usage-based",
            "      pricing reviews for high-spend customers to improve value perception.",
            "",
            "   6. FIBER OPTIC QUALITY AUDIT: Investigate service quality and pricing",
            "      competitiveness for fiber optic customers to address higher attrition.",
        ]
        for rec in recommendations:
            report_lines.append(rec)
        report_lines.append("")
        report_lines.append(separator)
        report_lines.append("  END OF REPORT")
        report_lines.append(separator)

        # ── Write to file ────────────────────────────────────────────────────
        report_text: str = "\n".join(report_lines)
        report_path: Path = REPORT_DIR / "business_insights.txt"
        report_path.write_text(report_text, encoding="utf-8")

        logger.info("Business insights report saved: %s", report_path)
        print(f"[SAVED] Business insights report → {report_path}")
        print(f"[INFO] Report size: {len(report_text):,} characters, {len(report_lines)} lines")
    except Exception as exc:
        logger.error("Failed to generate business insights report: %s", exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point for the EDA analysis pipeline.

    Orchestrates the complete pipeline:
        1. Load dataset.
        2. Profile dataset.
        3. Clean & preprocess data.
        4. Generate all visualizations.
        5. Write business insights report.
        6. Save cleaned dataset for downstream training.
    """
    try:
        logger.info("=" * 70)
        logger.info("TELCO CUSTOMER CHURN — EDA ANALYSIS PIPELINE STARTED")
        logger.info("=" * 70)

        print("\n" + "█" * 70)
        print("  TELCO CUSTOMER CHURN — EXPLORATORY DATA ANALYSIS")
        print("  Author: Krishnaveni Mitukula")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("█" * 70 + "\n")

        # Step 1: Load
        df: pd.DataFrame = load_dataset(DATASET_PATH)

        # Step 2: Profile
        profile: Dict[str, Any] = profile_dataset(df)

        # Step 3: Clean
        df = run_data_cleaning_pipeline(df)

        # Step 4: Visualize
        generate_all_visualizations(df)

        # Step 5: Business Insights
        generate_business_insights_report(df)

        # Step 6: Save cleaned dataset for training pipeline
        cleaned_path: Path = DATA_DIR / "cleaned_churn_data.csv"
        df.to_csv(cleaned_path, index=False)
        logger.info("Cleaned dataset saved: %s", cleaned_path)
        print(f"\n[SAVED] Cleaned dataset → {cleaned_path}")

        # ── Final Summary ────────────────────────────────────────────────────
        print("\n" + "█" * 70)
        print("  EDA PIPELINE COMPLETED SUCCESSFULLY")
        print(f"  Dataset Shape (cleaned)  : {df.shape}")
        print(f"  Visualizations saved     : images/")
        print(f"  Business report saved    : reports/business_insights.txt")
        print(f"  Cleaned data saved       : data/cleaned_churn_data.csv")
        print("█" * 70 + "\n")

        logger.info("EDA pipeline completed successfully.")

    except FileNotFoundError as exc:
        logger.critical("PIPELINE ABORTED — Dataset not found: %s", exc)
        print(f"\n[CRITICAL] Pipeline aborted: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.critical("PIPELINE ABORTED — Unexpected error: %s", exc)
        print(f"\n[CRITICAL] Pipeline aborted: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
