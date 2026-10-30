"""
data_loader.py
--------------
Loads raw subscription data, validates structure, performs data cleaning,
detects missing/duplicate values, and saves the cleaned dataset separately
in data/processed/ without mutating raw files.
"""

import os
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "subscription_id",
    "user_id",
    "subscription_name",
    "category",
    "plan_tier",
    "monthly_fee",
    "billing_cycle",
    "payment_method",
    "auto_renew",
    "usage_frequency_per_month",
    "hours_used_per_month",
    "days_since_last_login",
    "support_tickets_last_3m",
    "number_of_profiles_or_devices",
    "subscription_age_months",
]

def load_raw_data(filepath: str = "data/raw/subscriptions_raw.csv") -> pd.DataFrame:
    """
    Load raw subscription CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to raw CSV file.
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at '{filepath}'. Please generate or provide it.")
    
    logger.info(f"Loading raw data from '{filepath}'...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that required columns and basic data types are present.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate.
        
    Returns:
    --------
    Dict[str, Any]
        Validation summary report.
    """
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing mandatory columns in dataset: {missing_cols}")
    
    null_counts = df.isnull().sum().to_dict()
    total_nulls = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated(subset=["subscription_id"]).sum())
    
    report = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "total_nulls": total_nulls,
        "null_counts": null_counts,
        "duplicate_subscriptions": duplicate_count,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    logger.info(f"Data Schema Validation: {len(df)} rows, {total_nulls} nulls, {duplicate_count} duplicates.")
    return report

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataset:
    - Deduplicate based on subscription_id
    - Handle null values (imputation or removal)
    - Enforce logical boundaries (e.g., non-negative fees, non-negative usage)
    - Strip whitespace from strings
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input raw DataFrame.
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame.
    """
    cleaned = df.copy()
    
    # 1. Deduplication
    initial_len = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["subscription_id"])
    dups_removed = initial_len - len(cleaned)
    if dups_removed > 0:
        logger.info(f"Removed {dups_removed} duplicate records.")
        
    # 2. String trimming
    str_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        
    # 3. Numeric bounds validation
    # Monthly fee must be positive
    cleaned = cleaned[cleaned["monthly_fee"] > 0]
    
    # Usage days must be in range [0, 31]
    cleaned["usage_frequency_per_month"] = cleaned["usage_frequency_per_month"].clip(lower=0, upper=31)
    
    # Inactivity days non-negative
    cleaned["days_since_last_login"] = cleaned["days_since_last_login"].clip(lower=0)
    
    # Hours used non-negative
    cleaned["hours_used_per_month"] = cleaned["hours_used_per_month"].clip(lower=0.0)
    
    # Auto renew must be binary 0 or 1
    cleaned["auto_renew"] = cleaned["auto_renew"].astype(int).clip(lower=0, upper=1)
    
    # 4. Handle missing values if any exist
    if cleaned.isnull().values.any():
        logger.warning("Missing values found in data. Imputing median for numeric and mode for categorical.")
        for col in cleaned.columns:
            if cleaned[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                else:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
                    
    logger.info(f"Cleaned dataset has {len(cleaned)} rows and {len(cleaned.columns)} columns.")
    return cleaned

def load_and_process(
    raw_path: str = "data/raw/subscriptions_raw.csv",
    processed_path: str = "data/processed/subscriptions_cleaned.csv"
) -> pd.DataFrame:
    """
    Complete pipeline to load, validate, clean, and persist the processed data.
    """
    df_raw = load_raw_data(raw_path)
    validate_schema(df_raw)
    df_clean = clean_data(df_raw)
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    logger.info(f"Saved cleaned data to '{processed_path}'.")
    return df_clean

if __name__ == "__main__":
    df = load_and_process()
    print("Cleaned Data Preview:")
    print(df.head(3))
