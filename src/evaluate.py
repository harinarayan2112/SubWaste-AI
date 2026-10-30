"""
evaluate.py
-----------
Comprehensive model evaluation, error analysis, and visualization generator.

Key ML Concepts Demonstrated:
1. Metric Granularity for Asymmetric Business Costs:
   In subscription waste detection, False Negatives (missing a wasteful subscription)
   cause users to continue losing money quietly. False Positives (flagging an active
   service as wasteful) might mildly annoy the user. Hence, Recall and F1-Score are
   more critical than raw Accuracy.
2. Confusion Matrix Diagnostics:
   Visualizing True Positives, False Positives, True Negatives, and False Negatives.
3. Feature Importance Interpretation:
   Analyzing which behavioral/financial signals most heavily govern the model's decisions.
4. Error Interpretation & Limitations:
   Honest assessment of boundary errors, label noise, and production caveats.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from src.features import build_feature_dataset
from src.preprocessing import get_feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_confusion_matrix_plot(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "reports/confusion_matrix.png"
) -> np.ndarray:
    """
    Plot and save confusion matrix heatmap.
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Utilized (0)", "Wasteful (1)"],
        yticklabels=["Utilized (0)", "Wasteful (1)"],
        cbar=False
    )
    plt.title("SubWaste AI — Confusion Matrix (Held-out Test Set)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=10, labelpad=8)
    plt.ylabel("Actual True Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to '{output_path}'.")
    return cm

def generate_feature_importance_plot(
    pipeline,
    top_n: int = 12,
    output_path: str = "reports/feature_importance.png"
) -> pd.DataFrame:
    """
    Extract and plot top-N feature importances from the Random Forest classifier.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    feature_names = get_feature_names(preprocessor)
    importances = classifier.feature_importances_
    
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    top_df = df_imp.head(top_n)
    
    plt.figure(figsize=(9, 6))
    colors = sns.color_palette("viridis", len(top_df))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1], color=colors)
    plt.title(f"SubWaste AI — Top {top_n} Predictive Feature Importances", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Gini Importance Score", fontsize=10, labelpad=8)
    plt.ylabel("Feature Name", fontsize=10, labelpad=8)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance plot to '{output_path}'.")
    return df_imp

def evaluate_pipeline(
    model_path: str = "models/subwaste_pipeline.joblib",
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Run full evaluation suite on the saved pipeline using test split.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at '{model_path}'. Run training first.")
        
    pipeline = joblib.load(model_path)
    
    # Reload dataset and get test split (using same random state and stratify)
    X, y = build_feature_dataset()
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Classification Report
    clf_report_dict = classification_report(y_test, y_pred, output_dict=True)
    clf_report_text = classification_report(y_test, y_pred, target_names=["Utilized (0)", "Wasteful (1)"])
    
    # Confusion Matrix
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    cm = generate_confusion_matrix_plot(y_test, y_pred, cm_path)
    
    # Feature Importances
    imp_path = os.path.join(reports_dir, "feature_importance.png")
    df_imp = generate_feature_importance_plot(pipeline, top_n=12, output_path=imp_path)
    
    # Save detailed evaluation text report
    eval_text_path = os.path.join(reports_dir, "evaluation_report.txt")
    with open(eval_text_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SUBWASTE AI — HELD-OUT TEST EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(clf_report_text)
        f.write("\n\nConfusion Matrix:\n")
        f.write(f"TN: {cm[0, 0]} | FP: {cm[0, 1]}\n")
        f.write(f"FN: {cm[1, 0]} | TP: {cm[1, 1]}\n\n")
        f.write("Top 5 Drivers of Subscription Waste:\n")
        for idx, row in df_imp.head(5).iterrows():
            f.write(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("ERROR INTERPRETATION & LIMITATIONS:\n")
        f.write("1. False Positives (FP): Occur when users have high inactivity on services\n")
        f.write("   that require infrequent access (e.g. cloud backup, tax software) but which\n")
        f.write("   the user actively values. In production, category-specific thresholds or\n")
        f.write("   explicit user feedback should calibrate these.\n")
        f.write("2. False Negatives (FN): Occur on borderline usage (e.g. 1 login/month for 15 min)\n")
        f.write("   where unit cost is high but stochastic noise flipped the label.\n")
        f.write("3. Synthetic Caveat: High benchmark scores reflect clean underlying distributions.\n")
        f.write("   Real production deployment requires real banking/API telemetry and continuous drift monitoring.\n")
        f.write("=" * 60 + "\n")
        
    logger.info(f"Saved full evaluation report to '{eval_text_path}'.")
    
    return {
        "classification_report": clf_report_dict,
        "confusion_matrix": cm.tolist(),
        "top_features": df_imp.head(5).to_dict(orient="records")
    }

if __name__ == "__main__":
    results = evaluate_pipeline()
    print("\n--- Model Evaluation Completed ---")
    print("Confusion Matrix (TN, FP / FN, TP):", results["confusion_matrix"])
    print("\nTop 5 Features by Importance:")
    for feat in results["top_features"]:
        print(f" - {feat['feature']}: {feat['importance']:.4f}")
