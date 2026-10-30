"""
preprocessing.py
----------------
Defines scikit-learn preprocessing pipelines using ColumnTransformer.

Key ML Concepts Demonstrated:
1. Data Leakage Prevention:
   Scalers and encoders must be fit ONLY on training splits, then applied
   to test/inference splits using .transform().
2. Heterogeneous Feature Handling:
   Numerical features are standardized (zero mean, unit variance).
   Categorical features are One-Hot Encoded with `handle_unknown='ignore'`
   to prevent inference failures on unseen categories.
3. Clean Pipeline Modularity:
   Wrapping preprocessors inside standard scikit-learn abstractions enables
   seamless serialization, validation, and production inference.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.features import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def create_preprocessor(
    num_features: List[str] = NUMERICAL_FEATURES,
    cat_features: List[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Construct a ColumnTransformer that handles numerical scaling and categorical encoding.
    
    Parameters:
    -----------
    num_features : List[str]
        List of numerical feature names.
    cat_features : List[str]
        List of categorical feature names.
        
    Returns:
    --------
    ColumnTransformer
        Unfitted preprocessor transformer.
    """
    # Numerical pipeline: standard z-score scaling
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline: one-hot encoding with ignore for unseen categories
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop" # Drop any unmodeled identifiers like subscription_id
    )
    
    return preprocessor

def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieve feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    
    # Numerical features
    num_cols = preprocessor.transformers_[0][2]
    feature_names.extend(num_cols)
    
    # Categorical one-hot encoded features
    cat_transformer = preprocessor.transformers_[1][1]
    cat_cols = preprocessor.transformers_[1][2]
    onehot_encoder = cat_transformer.named_steps["onehot"]
    
    cat_feature_names = onehot_encoder.get_feature_names_out(cat_cols)
    feature_names.extend(list(cat_feature_names))
    
    return feature_names

if __name__ == "__main__":
    from src.features import build_feature_dataset
    X, y = build_feature_dataset()
    preprocessor = create_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    names = get_feature_names(preprocessor)
    print("Transformed Feature Matrix Shape:", X_trans.shape)
    print(f"Total Features after One-Hot Encoding: {len(names)}")
    print("First 10 Transformed Feature Names:", names[:10])
