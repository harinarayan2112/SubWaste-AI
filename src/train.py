"""
train.py
--------
Training pipeline for SubWaste AI.

Key ML Concepts Demonstrated:
1. Stratified Train-Test Splitting:
   Preserves target class distribution in both splits, preventing bias when
   wasteful subscriptions represent a minority class.
2. Leak-Free Pipeline Execution:
   Preprocessing and modeling are packaged into a unified scikit-learn Pipeline.
   All scaling and encoding statistics are calculated strictly on X_train.
3. Baseline Comparison:
   A simpler linear baseline (Logistic Regression) is compared against an ensemble
   method (Random Forest) to determine if non-linear interactions improve decision boundaries.
4. Reproducibility & Model Versioning:
   All stochastic operations use fixed seeds (`random_state=42`), and fitted pipelines
   are saved along with detailed metadata.
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
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.features import build_feature_dataset, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import create_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute core classification metrics.
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }

def train_and_compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Pipeline, Dict[str, Any], Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Execute stratified split, train baseline and Random Forest models,
    and return the champion model with full evaluation comparison.
    """
    logger.info("Splitting dataset into stratified train and test partitions...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples.")
    logger.info(f"Train waste rate: {y_train.mean():.2%}, Test waste rate: {y_test.mean():.2%}")
    
    # Preprocessing
    preprocessor = create_preprocessor()
    
    # 1. Baseline Model: Logistic Regression
    logger.info("Training Baseline Model: Logistic Regression (balanced class weights)...")
    baseline_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=random_state, class_weight="balanced", max_iter=1000))
    ])
    baseline_pipeline.fit(X_train, y_train)
    
    y_pred_base = baseline_pipeline.predict(X_test)
    y_proba_base = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_predictions(y_test, y_pred_base, y_proba_base)
    logger.info(f"Baseline Results: F1={baseline_metrics['f1_score']}, Recall={baseline_metrics['recall']}, Precision={baseline_metrics['precision']}")
    
    # 2. Advanced Champion Model: Random Forest Classifier
    logger.info("Training Advanced Model: Random Forest Classifier...")
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    
    y_pred_rf = rf_pipeline.predict(X_test)
    y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_predictions(y_test, y_pred_rf, y_proba_rf)
    logger.info(f"Random Forest Results: F1={rf_metrics['f1_score']}, Recall={rf_metrics['recall']}, Precision={rf_metrics['precision']}")
    
    # Comparison summary
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models": {
            "Baseline_LogisticRegression": baseline_metrics,
            "Champion_RandomForest": rf_metrics
        },
        "chosen_model": "Champion_RandomForest"
    }
    
    # We select the Random Forest pipeline as champion
    return rf_pipeline, comparison, (X_train, X_test, y_train, y_test)

def save_pipeline_artifacts(
    pipeline: Pipeline,
    comparison: Dict[str, Any],
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Tuple[str, str]:
    """
    Serialize the fitted model pipeline and save comparison reports.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "subwaste_pipeline.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    report_path = os.path.join(reports_dir, "model_comparison.json")
    
    logger.info(f"Saving serialized pipeline to '{model_path}'...")
    joblib.dump(pipeline, model_path)
    
    metadata = {
        "model_type": "RandomForestClassifier",
        "pipeline_steps": list(pipeline.named_steps.keys()),
        "hyperparameters": pipeline.named_steps["classifier"].get_params(),
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "evaluation_metrics": comparison["models"]["Champion_RandomForest"],
        "saved_at": datetime.now().isoformat()
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=4)
        
    logger.info("Pipeline and metadata artifacts successfully saved.")
    return model_path, metadata_path

def run_training_pipeline() -> Tuple[Pipeline, Dict[str, Any], Tuple]:
    """
    High-level orchestrator function to build features, train, evaluate, and save pipeline.
    """
    X, y = build_feature_dataset()
    champion_pipeline, comparison, splits = train_and_compare_models(X, y)
    save_pipeline_artifacts(champion_pipeline, comparison)
    return champion_pipeline, comparison, splits

if __name__ == "__main__":
    pipeline, comparison, splits = run_training_pipeline()
    print("\n--- Model Training Summary ---")
    print(json.dumps(comparison, indent=2))
