"""
test_pipeline.py
----------------
Unit and integration tests for the SubWaste AI ML pipeline.
Tests data loading, schema validation, feature engineering, pipeline inference,
model persistence, and defensive input validation.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.data_loader import load_raw_data, validate_schema, clean_data
from src.features import engineer_features, define_target, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import create_preprocessor
from src.train import run_training_pipeline
from src.predict import SubWastePredictor

class TestSubWastePipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Prepare sample test data and ensure pipeline is trained."""
        cls.raw_path = "data/raw/subscriptions_raw.csv"
        if not os.path.exists(cls.raw_path):
            from scripts.generate_raw_data import generate_synthetic_subscriptions
            os.makedirs("data/raw", exist_ok=True)
            df = generate_synthetic_subscriptions(num_samples=200, random_state=42)
            df.to_csv(cls.raw_path, index=False)
            
        cls.df_raw = pd.read_csv(cls.raw_path)
        cls.sample_record = {
            "subscription_name": "Test Stream",
            "category": "Streaming",
            "plan_tier": "Standard",
            "monthly_fee": 14.99,
            "billing_cycle": "Monthly",
            "payment_method": "Credit Card",
            "auto_renew": 1,
            "usage_frequency_per_month": 20,
            "hours_used_per_month": 35.0,
            "days_since_last_login": 2,
            "support_tickets_last_3m": 0,
            "number_of_profiles_or_devices": 2,
            "subscription_age_months": 8
        }
        
    def test_01_schema_validation(self):
        """Test that schema validator passes on valid raw data and fails on missing columns."""
        report = validate_schema(self.df_raw)
        self.assertIn("num_rows", report)
        self.assertEqual(report["duplicate_subscriptions"], 0)
        
        # Corrupt dataframe by dropping a column
        bad_df = self.df_raw.drop(columns=["monthly_fee"])
        with self.assertRaises(ValueError):
            validate_schema(bad_df)
            
    def test_02_data_cleaning(self):
        """Verify cleaning bounds and null handling."""
        # Inject an out-of-bounds negative fee and duplicate
        dirty_df = self.df_raw.copy()
        dirty_df.loc[0, "monthly_fee"] = -10.0
        dirty_df = pd.concat([dirty_df, dirty_df.iloc[[1]]]) # duplicate
        
        cleaned = clean_data(dirty_df)
        self.assertTrue((cleaned["monthly_fee"] > 0).all())
        self.assertEqual(cleaned.duplicated(subset=["subscription_id"]).sum(), 0)
        
    def test_03_feature_engineering(self):
        """Verify feature engineering produces expected ratios without ZeroDivisionError."""
        featured = engineer_features(self.df_raw)
        
        self.assertIn("cost_per_active_hour", featured.columns)
        self.assertIn("inactivity_ratio", featured.columns)
        self.assertIn("utilization_rate", featured.columns)
        self.assertIn("fee_to_device_ratio", featured.columns)
        
        # Check no infinite or NaN values
        self.assertFalse(np.isinf(featured["cost_per_active_hour"]).any())
        self.assertFalse(featured["cost_per_active_hour"].isnull().any())
        
    def test_04_target_generation(self):
        """Verify target is strictly binary and distributed sensibly."""
        featured = engineer_features(self.df_raw)
        target = define_target(featured)
        
        self.assertEqual(set(target.unique()), {0, 1})
        # Check waste rate is within realistic range (e.g. 15% to 50%)
        waste_rate = target.mean()
        self.assertGreater(waste_rate, 0.15)
        self.assertLess(waste_rate, 0.55)
        
    def test_05_preprocessor_column_transformer(self):
        """Verify ColumnTransformer correctly scales numerics and one-hot encodes categoricals."""
        featured = engineer_features(self.df_raw)
        X = featured[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        preprocessor = create_preprocessor()
        
        X_trans = preprocessor.fit_transform(X)
        self.assertEqual(X_trans.shape[0], len(X))
        # Total columns must be greater than raw feature count due to one-hot encoding
        self.assertGreater(X_trans.shape[1], len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES))
        
    def test_06_model_inference_and_validation(self):
        """Verify trained predictor output, accuracy, and input validation."""
        # Ensure model is trained and saved
        if not os.path.exists("models/subwaste_pipeline.joblib"):
            run_training_pipeline()
            
        predictor = SubWastePredictor("models/subwaste_pipeline.joblib")
        
        # Predict active record
        res_active = predictor.predict_single(self.sample_record)
        self.assertEqual(res_active["is_wasteful"], 0)
        self.assertLess(res_active["waste_probability"], 50.0)
        
        # Predict dormant/wasteful record
        wasteful_record = self.sample_record.copy()
        wasteful_record["days_since_last_login"] = 60
        wasteful_record["usage_frequency_per_month"] = 0
        wasteful_record["hours_used_per_month"] = 0.0
        wasteful_record["monthly_fee"] = 50.00
        wasteful_record["plan_tier"] = "Premium"
        
        res_wasteful = predictor.predict_single(wasteful_record)
        self.assertEqual(res_wasteful["is_wasteful"], 1)
        self.assertGreater(res_wasteful["waste_probability"], 60.0)
        self.assertIn("Cancel or Pause", res_wasteful["recommendation"])
        
        # Test validation error on invalid input
        invalid_record = self.sample_record.copy()
        invalid_record["monthly_fee"] = -5.0
        with self.assertRaises(ValueError):
            predictor.predict_single(invalid_record)

if __name__ == "__main__":
    unittest.main()
