"""
generate_raw_data.py
--------------------
Generates a realistic, reproducible synthetic raw dataset of subscription usage telemetry.

Why synthetic data is used here:
1. Public subscription datasets (e.g., Telco Churn, Netflix Kaggle) only track CUSTOMER CHURN
   (cancellation), not ACTIVE SUBSCRIPTION WASTEFULNESS (underutilization while recurring billing continues).
2. Conflating churn with wastefulness is a fundamental domain mistake. Churned users no longer pay.
   Wasteful users continue paying for services they do not use ("zombie subscriptions").
3. Ground-truth telemetry from real FinOps/personal finance apps is proprietary and subject to GDPR/privacy.
4. Hence, we construct a statistically sound, realistic benchmark dataset reflecting real-world usage patterns.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_subscriptions(num_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic subscription telemetry.
    
    Parameters:
    -----------
    num_samples : int
        Number of subscription records to generate.
    random_state : int
        Seed for reproducible pseudo-random generation.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with raw subscription records.
    """
    np.random.seed(random_state)
    
    # Common subscription archetypes
    catalog = [
        {"name": "Netflix", "category": "Streaming", "base_fee": 15.49, "tier_mult": {"Basic": 0.65, "Standard": 1.0, "Premium": 1.48}},
        {"name": "Spotify", "category": "Streaming", "base_fee": 10.99, "tier_mult": {"Basic": 0.6, "Standard": 1.0, "Premium": 1.55}},
        {"name": "Disney+", "category": "Streaming", "base_fee": 13.99, "tier_mult": {"Basic": 0.57, "Standard": 1.0, "Premium": 1.35}},
        {"name": "Adobe Creative Cloud", "category": "Productivity", "base_fee": 54.99, "tier_mult": {"Basic": 0.4, "Standard": 1.0, "Premium": 1.6}},
        {"name": "ChatGPT Plus", "category": "Productivity", "base_fee": 20.00, "tier_mult": {"Basic": 0.5, "Standard": 1.0, "Premium": 1.5}},
        {"name": "Google One / Drive", "category": "Cloud Storage", "base_fee": 9.99, "tier_mult": {"Basic": 0.2, "Standard": 1.0, "Premium": 2.5}},
        {"name": "iCloud Storage", "category": "Cloud Storage", "base_fee": 9.99, "tier_mult": {"Basic": 0.3, "Standard": 1.0, "Premium": 3.0}},
        {"name": "Gym Membership (Anytime/Gold)", "category": "Fitness", "base_fee": 49.99, "tier_mult": {"Basic": 0.6, "Standard": 1.0, "Premium": 1.8}},
        {"name": "Peloton Digital", "category": "Fitness", "base_fee": 24.00, "tier_mult": {"Basic": 0.5, "Standard": 1.0, "Premium": 1.7}},
        {"name": "Xbox Game Pass", "category": "Gaming", "base_fee": 16.99, "tier_mult": {"Basic": 0.7, "Standard": 1.0, "Premium": 1.3}},
        {"name": "PlayStation Plus", "category": "Gaming", "base_fee": 14.99, "tier_mult": {"Basic": 0.65, "Standard": 1.0, "Premium": 1.45}},
        {"name": "Coursera Plus", "category": "Education", "base_fee": 39.00, "tier_mult": {"Basic": 0.5, "Standard": 1.0, "Premium": 1.5}},
        {"name": "Duolingo Super", "category": "Education", "base_fee": 12.99, "tier_mult": {"Basic": 0.7, "Standard": 1.0, "Premium": 1.5}},
        {"name": "Microsoft 365", "category": "Productivity", "base_fee": 12.99, "tier_mult": {"Basic": 0.55, "Standard": 1.0, "Premium": 1.7}}
    ]
    
    tiers = ["Basic", "Standard", "Premium"]
    tier_weights = [0.35, 0.45, 0.20]
    
    billing_cycles = ["Monthly", "Annual"]
    billing_weights = [0.70, 0.30]
    
    payment_methods = ["Credit Card", "Debit Card", "PayPal", "Auto-Debit"]
    payment_weights = [0.45, 0.25, 0.20, 0.10]
    
    records = []
    
    for i in range(num_samples):
        sub_id = f"SUB-{10001 + i}"
        user_id = f"USR-{np.random.randint(1000, 4500)}"
        
        # Pick service
        service = np.random.choice(catalog)
        name = service["name"]
        cat = service["category"]
        
        plan_tier = np.random.choice(tiers, p=tier_weights)
        billing = np.random.choice(billing_cycles, p=billing_weights)
        payment = np.random.choice(payment_methods, p=payment_weights)
        
        # Calculate fee based on tier and slight variance
        base_fee = service["base_fee"] * service["tier_mult"][plan_tier]
        monthly_fee = round(base_fee * np.random.uniform(0.95, 1.05), 2)
        
        auto_renew = np.random.choice([1, 0], p=[0.88, 0.12])
        tenure_months = int(np.random.exponential(scale=14.0) + 1)
        tenure_months = min(max(tenure_months, 1), 60)
        
        # 3 latent user archetypes:
        # A) Heavy/Daily Active Users (~45%)
        # B) Moderate/Occasional Users (~25%)
        # C) Dormant/Forgotten/Wasteful Users (~30%)
        archetype = np.random.choice(["active", "moderate", "dormant"], p=[0.45, 0.25, 0.30])
        
        if archetype == "active":
            usage_days = int(np.random.randint(15, 30))
            hours_used = round(float(np.random.uniform(15.0, 75.0)), 1)
            days_inactive = int(np.random.exponential(scale=2.0))
            days_inactive = min(days_inactive, 8)
            tickets = int(np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05]))
            devices = int(np.random.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.25, 0.10]))
            
        elif archetype == "moderate":
            usage_days = int(np.random.randint(5, 16))
            hours_used = round(float(np.random.uniform(4.0, 20.0)), 1)
            days_inactive = int(np.random.randint(4, 20))
            tickets = int(np.random.choice([0, 1, 2], p=[0.80, 0.15, 0.05]))
            devices = int(np.random.choice([1, 2, 3], p=[0.40, 0.45, 0.15]))
            
        else: # dormant / underutilized
            usage_days = int(np.random.choice([0, 1, 2, 3], p=[0.55, 0.25, 0.15, 0.05]))
            hours_used = round(float(np.random.uniform(0.0, 2.5)), 1)
            days_inactive = int(np.random.randint(22, 90))
            tickets = int(np.random.choice([0, 1], p=[0.90, 0.10]))
            devices = 1 if np.random.rand() > 0.3 else 2
        
        # Add realistic noise/anomalies (e.g. edge cases)
        if np.random.rand() < 0.03:
            # edge case: sudden holiday spike or zero hours despite 1 day login
            hours_used = 0.0
            
        records.append({
            "subscription_id": sub_id,
            "user_id": user_id,
            "subscription_name": name,
            "category": cat,
            "plan_tier": plan_tier,
            "monthly_fee": monthly_fee,
            "billing_cycle": billing,
            "payment_method": payment,
            "auto_renew": auto_renew,
            "usage_frequency_per_month": usage_days,
            "hours_used_per_month": hours_used,
            "days_since_last_login": days_inactive,
            "support_tickets_last_3m": tickets,
            "number_of_profiles_or_devices": devices,
            "subscription_age_months": tenure_months
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/subscriptions_raw.csv"
    df = generate_synthetic_subscriptions(num_samples=1500, random_state=42)
    df.to_csv(out_path, index=False)
    print(f"Successfully generated {len(df)} synthetic subscription records at '{out_path}'.")
    print("Columns:", list(df.columns))
    print(df.head(3))
