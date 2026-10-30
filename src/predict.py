"""
predict.py
----------
Inference service and input validator for SubWaste AI.

Key ML Concepts Demonstrated:
1. Production Feature Pipeline Consistency:
   Raw user inputs must undergo the exact same feature engineering transformations
   as training data before hitting the model pipeline.
2. Defensive Input Validation:
   Production ML systems must reject malformed, out-of-range, or corrupted input
   payloads before inference rather than crashing inside matrix calculations.
3. Interpretable Output:
   Converts raw probability output into actionable business recommendations
   (e.g., Risk Level, Cost-per-hour penalty, suggested action).
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Dict, Any, List, Union
import joblib
import pandas as pd
import numpy as np

from src.features import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "Streaming", "Productivity", "Cloud Storage", "Fitness", "Gaming", "Education", "Utilities", "Other"
}
ALLOWED_TIERS = {"Basic", "Standard", "Premium"}
ALLOWED_CYCLES = {"Monthly", "Annual"}

class SubWastePredictor:
    """
    SubWaste AI Prediction Service.
    Loads fitted pipeline and performs safe inference on subscription payloads.
    """
    def __init__(self, model_path: str = "models/subwaste_pipeline.joblib"):
        self.model_path = model_path
        self.pipeline = self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. Please run 'python src/train.py' first."
            )
        logger.info(f"Loading trained pipeline from '{self.model_path}'...")
        return joblib.load(self.model_path)
    
    def validate_input(self, record: Dict[str, Any]) -> None:
        """
        Validate schema, data types, and value boundaries.
        Raises ValueError with clear message if invalid.
        """
        required_keys = [
            "subscription_name", "category", "plan_tier", "monthly_fee",
            "billing_cycle", "payment_method", "auto_renew",
            "usage_frequency_per_month", "hours_used_per_month",
            "days_since_last_login", "support_tickets_last_3m",
            "number_of_profiles_or_devices", "subscription_age_months"
        ]
        
        # Check missing keys
        missing = [k for k in required_keys if k not in record]
        if missing:
            raise ValueError(f"Input validation failed. Missing required fields: {missing}")
            
        # Range checks
        if record["monthly_fee"] <= 0:
            raise ValueError(f"Invalid 'monthly_fee': {record['monthly_fee']}. Must be strictly positive.")
            
        if not (0 <= record["usage_frequency_per_month"] <= 31):
            raise ValueError(
                f"Invalid 'usage_frequency_per_month': {record['usage_frequency_per_month']}. Must be between 0 and 31."
            )
            
        if record["hours_used_per_month"] < 0:
            raise ValueError(
                f"Invalid 'hours_used_per_month': {record['hours_used_per_month']}. Cannot be negative."
            )
            
        if record["days_since_last_login"] < 0:
            raise ValueError(
                f"Invalid 'days_since_last_login': {record['days_since_last_login']}. Cannot be negative."
            )
            
        if record["auto_renew"] not in (0, 1):
            raise ValueError(f"Invalid 'auto_renew': {record['auto_renew']}. Must be 0 or 1.")
            
        if record["plan_tier"] not in ALLOWED_TIERS:
            raise ValueError(f"Invalid 'plan_tier': {record['plan_tier']}. Allowed: {ALLOWED_TIERS}")
            
    def _generate_recommendation(self, record: Dict[str, Any], proba: float, is_wasteful: int) -> str:
        """
        Generate contextual, actionable user recommendations based on feature signals.
        """
        fee = record["monthly_fee"]
        hours = record["hours_used_per_month"]
        inactive = record["days_since_last_login"]
        sub_name = record["subscription_name"]
        
        cost_per_hr = fee / (hours + 0.1)
        annual_drain = fee * 12.0
        
        if is_wasteful == 1:
            if inactive >= 30:
                return (
                    f"Action: Cancel or Pause Immediately. You haven't logged in for {inactive} days "
                    f"while paying ${fee:.2f}/month. Canceling saves you ${annual_drain:.2f}/year."
                )
            elif cost_per_hr > 15.0:
                return (
                    f"Action: Downgrade or Cancel. You only used {hours:.1f} hours last month, costing you "
                    f"${cost_per_hr:.2f} per active hour. Consider a lower tier or pay-as-you-go."
                )
            else:
                return (
                    f"Action: Review Subscription. High risk of underutilization detected. "
                    f"Check if you still need {sub_name} for ${fee:.2f}/month."
                )
        else:
            if proba > 0.35:
                return f"Notice: Healthy usage, but watch out: inactivity is {inactive} days. Keep an eye on upcoming billing."
            return f"Status: Highly Utilized. You are getting strong economic value from {sub_name} (${cost_per_hr:.2f}/hr)."

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict wastefulness for a single subscription dictionary.
        """
        self.validate_input(record)
        
        # Convert to single-row dataframe
        df_raw = pd.DataFrame([record])
        df_featured = engineer_features(df_raw)
        
        # Select modeled features
        feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        X = df_featured[feature_cols]
        
        # Inference
        pred_class = int(self.pipeline.predict(X)[0])
        pred_proba = float(self.pipeline.predict_proba(X)[0][1])
        
        # Risk level
        if pred_proba >= 0.70:
            risk_level = "HIGH RISK (Wasteful)"
        elif pred_proba >= 0.40:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "LOW RISK (Well Utilized)"
            
        recommendation = self._generate_recommendation(record, pred_proba, pred_class)
        
        return {
            "subscription_name": record["subscription_name"],
            "monthly_fee": record["monthly_fee"],
            "is_wasteful": pred_class,
            "waste_probability": round(pred_proba * 100, 2),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "engineered_metrics": {
                "cost_per_active_hour": float(df_featured["cost_per_active_hour"].iloc[0]),
                "inactivity_ratio": float(df_featured["inactivity_ratio"].iloc[0]),
                "annual_projected_cost": float(df_featured["annual_projected_cost"].iloc[0]),
            }
        }
        
    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict wastefulness for multiple subscriptions.
        """
        return [self.predict_single(r) for r in records]

if __name__ == "__main__":
    predictor = SubWastePredictor()
    
    # Test Sample 1: Forgotten high-cost software
    sample_wasteful = {
        "subscription_name": "Adobe Creative Cloud",
        "category": "Productivity",
        "plan_tier": "Premium",
        "monthly_fee": 54.99,
        "billing_cycle": "Monthly",
        "payment_method": "Credit Card",
        "auto_renew": 1,
        "usage_frequency_per_month": 0,
        "hours_used_per_month": 0.0,
        "days_since_last_login": 45,
        "support_tickets_last_3m": 0,
        "number_of_profiles_or_devices": 1,
        "subscription_age_months": 18
    }
    
    # Test Sample 2: Daily active streaming
    sample_active = {
        "subscription_name": "Spotify",
        "category": "Streaming",
        "plan_tier": "Standard",
        "monthly_fee": 10.99,
        "billing_cycle": "Monthly",
        "payment_method": "Debit Card",
        "auto_renew": 1,
        "usage_frequency_per_month": 24,
        "hours_used_per_month": 45.0,
        "days_since_last_login": 1,
        "support_tickets_last_3m": 0,
        "number_of_profiles_or_devices": 2,
        "subscription_age_months": 12
    }
    
    print("\n--- SAMPLE 1: DORMANT HIGH-COST SUBSCRIPTION ---")
    res1 = predictor.predict_single(sample_wasteful)
    print(res1)
    
    print("\n--- SAMPLE 2: ACTIVE DAILY SUBSCRIPTION ---")
    res2 = predictor.predict_single(sample_active)
    print(res2)
    
    print("\n--- TESTING INPUT VALIDATION (NEGATIVE FEE) ---")
    try:
        sample_invalid = sample_active.copy()
        sample_invalid["monthly_fee"] = -15.00
        predictor.predict_single(sample_invalid)
    except ValueError as e:
        print(f"Defensive check successfully caught invalid record: {e}")
