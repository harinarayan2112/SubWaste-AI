"""
features.py
-----------
Feature engineering and target definition for SubWaste AI.

Key ML Concepts Demonstrated:
1. Domain-Driven Feature Engineering:
   Deriving ratio and interaction features (e.g., cost per active hour, inactivity ratio)
   that translate raw telemetry into meaningful economic utility signals.
2. Defensible Target Strategy:
   Formulating a multi-criteria utility heuristic to label underutilized subscriptions,
   while incorporating realistic noise and boundary conditions.
3. Separation of Concerns:
   Raw features are transformed in a vectorized, leakage-free manner.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Tuple, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature Column Definitions
NUMERICAL_FEATURES = [
    "monthly_fee",
    "usage_frequency_per_month",
    "hours_used_per_month",
    "days_since_last_login",
    "support_tickets_last_3m",
    "number_of_profiles_or_devices",
    "subscription_age_months",
    "cost_per_active_hour",
    "cost_per_active_day",
    "inactivity_ratio",
    "utilization_rate",
    "annual_projected_cost",
    "fee_to_device_ratio",
]

CATEGORICAL_FEATURES = [
    "category",
    "plan_tier",
    "billing_cycle",
    "payment_method",
    "auto_renew",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived financial and behavioral features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataframe containing raw subscription columns.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added engineered features.
    """
    data = df.copy()
    
    # 1. Cost per active hour (smoothed with +0.1 to prevent ZeroDivisionError)
    data["cost_per_active_hour"] = np.round(
        data["monthly_fee"] / (data["hours_used_per_month"] + 0.1), 2
    )
    
    # 2. Cost per active day
    data["cost_per_active_day"] = np.round(
        data["monthly_fee"] / (data["usage_frequency_per_month"] + 0.1), 2
    )
    
    # 3. Inactivity ratio relative to a standard 30-day billing cycle
    # Values > 1.0 indicate the user went an entire billing period without logging in.
    data["inactivity_ratio"] = np.round(data["days_since_last_login"] / 30.0, 3)
    
    # 4. Utilization rate: fraction of days active in the month
    data["utilization_rate"] = np.round(data["usage_frequency_per_month"] / 30.0, 3)
    
    # 5. Annual projected cost
    data["annual_projected_cost"] = np.round(data["monthly_fee"] * 12.0, 2)
    
    # 6. Fee to device/profile ratio (identifies single users paying for family/multi-user tiers)
    devices = data["number_of_profiles_or_devices"].clip(lower=1)
    data["fee_to_device_ratio"] = np.round(data["monthly_fee"] / devices, 2)
    
    return data

def define_target(df: pd.DataFrame, noise_rate: float = 0.08, random_state: int = 42) -> pd.Series:
    """
    Defines the ground-truth target variable 'is_wasteful'.
    
    Methodology and Assumptions:
    -----------------------------
    A subscription is labeled as 'wasteful' (1) if it exhibits poor economic utility:
      Condition 1 (Severe Inactivity):
        Inactivity >= 30 days AND active usage days <= 1 AND auto-renew is ON.
      Condition 2 (Extreme Unit Cost):
        Cost per active hour > $12.00 AND total monthly usage < 2.5 hours.
      Condition 3 (High-Tier Zombie Spend):
        Plan is 'Premium' AND hours used < 2.0 AND monthly fee >= $18.00.
        
    Noise Injection:
      Real user value is partially subjective (e.g., someone paying for cloud storage
      or password managers that don't require frequent logins). We inject an 8% label
      perturbation to model real-world uncertainty and evaluate generalization.
      
    Returns:
    --------
    pd.Series
        Binary series (1 = wasteful, 0 = utilized).
    """
    np.random.seed(random_state)
    
    # Criteria evaluation
    cond1 = (df["days_since_last_login"] >= 30) & (df["usage_frequency_per_month"] <= 1) & (df["auto_renew"] == 1)
    cond2 = (df["cost_per_active_hour"] > 12.0) & (df["hours_used_per_month"] < 2.5)
    cond3 = (df["plan_tier"] == "Premium") & (df["hours_used_per_month"] < 2.0) & (df["monthly_fee"] >= 18.0)
    
    # Ground truth heuristic
    wasteful_rule = (cond1 | cond2 | cond3).astype(int)
    
    # Stochastic noise injection
    noise_mask = np.random.rand(len(df)) < noise_rate
    target = wasteful_rule.copy()
    target[noise_mask] = 1 - target[noise_mask]
    
    waste_rate = target.mean() * 100
    logger.info(f"Target defined: {int(target.sum())}/{len(target)} labeled as wasteful ({waste_rate:.1f}%).")
    return target

def build_feature_dataset(
    input_path: str = "data/processed/subscriptions_cleaned.csv"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load cleaned data, perform feature engineering, and attach target.
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        X (Feature matrix), y (Target vector)
    """
    df = pd.read_csv(input_path)
    df_features = engineer_features(df)
    
    # Target
    y = define_target(df_features)
    
    # Feature matrix X: select only modeled features
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df_features[feature_cols].copy()
    
    return X, y

if __name__ == "__main__":
    X, y = build_feature_dataset()
    print("Features Shape:", X.shape)
    print("Target Distribution:\n", y.value_counts(normalize=True))
    print("\nSample Engineered Features:")
    print(X[NUMERICAL_FEATURES[:5]].head(3))
