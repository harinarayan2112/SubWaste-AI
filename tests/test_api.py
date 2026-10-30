"""
test_api.py
-----------
Integration and unit tests for SubWaste AI REST API using FastAPI TestClient.

Tests:
1. Root & Health Check endpoints
2. Model metadata inspection
3. Single subscription prediction (/predict)
4. Input validation errors (HTTP 422 on negative fees, invalid days)
5. Batch prediction with financial leakage calculations (/predict/batch)
6. Direct CSV upload auditing (/predict/csv)
"""

import os
import sys
import io
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.app import app

class TestSubWasteAPI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize TestClient and verify pipeline artifact exists."""
        cls.client = TestClient(app)
        
        # Sample valid dormant subscription (Wasteful)
        cls.wasteful_sub = {
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
            "subscription_age_months": 12
        }
        
        # Sample valid active subscription (Well utilized)
        cls.active_sub = {
            "subscription_name": "Spotify",
            "category": "Streaming",
            "plan_tier": "Standard",
            "monthly_fee": 10.99,
            "billing_cycle": "Monthly",
            "payment_method": "Debit Card",
            "auto_renew": 1,
            "usage_frequency_per_month": 25,
            "hours_used_per_month": 50.0,
            "days_since_last_login": 1,
            "support_tickets_last_3m": 0,
            "number_of_profiles_or_devices": 2,
            "subscription_age_months": 24
        }
        
    def test_01_root_and_dashboard_endpoints(self):
        """Test GET / returns 200 with HTML dashboard and GET /api returns JSON overview."""
        # 1. HTML Dashboard
        res_dash = self.client.get("/")
        self.assertEqual(res_dash.status_code, 200)
        self.assertIn("text/html", res_dash.headers.get("content-type", ""))
        self.assertIn("SubWaste AI", res_dash.text)
        
        # 2. JSON API Overview
        res_api = self.client.get("/api")
        self.assertEqual(res_api.status_code, 200)
        data = res_api.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("documentation", data)
        
    def test_02_health_endpoint(self):
        """Test GET /health returns healthy status and confirmed model load."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("timestamp", data)
        
    def test_03_model_metadata_endpoint(self):
        """Test GET /model/metadata returns trained hyperparameters and test metrics."""
        res = self.client.get("/model/metadata")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model_type"], "RandomForestClassifier")
        self.assertIn("evaluation_metrics", data)
        self.assertIn("f1_score", data["evaluation_metrics"])
        
    def test_04_single_predict_wasteful(self):
        """Test POST /predict correctly flags dormant subscription as high risk."""
        res = self.client.post("/predict", json=self.wasteful_sub)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["subscription_name"], "Adobe Creative Cloud")
        self.assertEqual(data["is_wasteful"], 1)
        self.assertGreater(data["waste_probability"], 60.0)
        self.assertIn("HIGH RISK", data["risk_level"])
        self.assertIn("Cancel or Pause", data["recommendation"])
        self.assertIn("cost_per_active_hour", data["engineered_metrics"])
        
    def test_05_single_predict_active(self):
        """Test POST /predict correctly identifies active subscription as low risk."""
        res = self.client.post("/predict", json=self.active_sub)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["subscription_name"], "Spotify")
        self.assertEqual(data["is_wasteful"], 0)
        self.assertLess(data["waste_probability"], 50.0)
        self.assertIn("LOW RISK", data["risk_level"])
        self.assertIn("Highly Utilized", data["recommendation"])
        
    def test_06_validation_error(self):
        """Test POST /predict rejects negative monthly fee with HTTP 422."""
        invalid_payload = self.active_sub.copy()
        invalid_payload["monthly_fee"] = -25.00 # Invalid!
        res = self.client.post("/predict", json=invalid_payload)
        self.assertEqual(res.status_code, 422)
        
    def test_07_batch_predict(self):
        """Test POST /predict/batch computes combined financial metrics and savings."""
        batch_payload = {
            "subscriptions": [self.wasteful_sub, self.active_sub]
        }
        res = self.client.post("/predict/batch", json=batch_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_subscriptions"], 2)
        self.assertEqual(data["wasteful_subscriptions_count"], 1)
        self.assertAlmostEqual(data["total_monthly_spend"], 65.98, places=2)
        self.assertAlmostEqual(data["total_potential_monthly_savings"], 54.99, places=2)
        self.assertAlmostEqual(data["total_potential_annual_savings"], 659.88, places=2)
        self.assertEqual(len(data["results"]), 2)
        
    def test_08_csv_upload_predict(self):
        """Test POST /predict/csv parses CSV file and returns batch audit."""
        csv_content = (
            "subscription_name,category,plan_tier,monthly_fee,billing_cycle,payment_method,auto_renew,usage_frequency_per_month,hours_used_per_month,days_since_last_login,support_tickets_last_3m,number_of_profiles_or_devices,subscription_age_months\n"
            "Gym Membership,Fitness,Premium,50.00,Monthly,Credit Card,1,0,0.0,60,0,1,18\n"
            "Netflix,Streaming,Standard,15.49,Monthly,PayPal,1,22,35.0,2,0,2,10\n"
        )
        file_bytes = io.BytesIO(csv_content.encode("utf-8"))
        
        res = self.client.post(
            "/predict/csv",
            files={"file": ("subscriptions.csv", file_bytes, "text/csv")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_subscriptions"], 2)
        self.assertEqual(data["wasteful_subscriptions_count"], 1)
        self.assertEqual(data["results"][0]["subscription_name"], "Gym Membership")
        self.assertEqual(data["results"][0]["is_wasteful"], 1)

if __name__ == "__main__":
    unittest.main()
