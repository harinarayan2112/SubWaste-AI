"""
schemas.py
----------
Pydantic v2 schemas for SubWaste AI REST API.

Key ML & Backend Concepts Demonstrated:
1. Strict Type Validation & Data Contracts:
   Validates payloads BEFORE reaching the ML pipeline. Invalid values (e.g. negative fees,
   out-of-bounds days) are caught at the HTTP gateway with clear 422 Unprocessable Entity errors.
2. OpenAPI Documentation with Schemas & Examples:
   Provides rich field descriptions and realistic schema examples so the interactive
   Swagger UI (/docs) is immediately functional and self-documenting.
3. Separation of Input, Output, and Batch Aggregates:
   Encapsulates financial metrics (annual savings, monthly leak) alongside model risk predictions.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class PlanTier(str, Enum):
    BASIC = "Basic"
    STANDARD = "Standard"
    PREMIUM = "Premium"

class BillingCycle(str, Enum):
    MONTHLY = "Monthly"
    ANNUAL = "Annual"

class SubscriptionCategory(str, Enum):
    STREAMING = "Streaming"
    PRODUCTIVITY = "Productivity"
    CLOUD_STORAGE = "Cloud Storage"
    FITNESS = "Fitness"
    GAMING = "Gaming"
    EDUCATION = "Education"
    UTILITIES = "Utilities"
    OTHER = "Other"

class SubscriptionInput(BaseModel):
    """Input payload for a single subscription evaluation."""
    subscription_name: str = Field(..., min_length=1, max_length=100, description="Name of the service (e.g. Netflix, Adobe CC)")
    category: SubscriptionCategory = Field(default=SubscriptionCategory.STREAMING, description="Subscription service category")
    plan_tier: PlanTier = Field(default=PlanTier.STANDARD, description="Pricing tier")
    monthly_fee: float = Field(..., gt=0.0, description="Recurring monthly fee in USD (must be positive)")
    billing_cycle: BillingCycle = Field(default=BillingCycle.MONTHLY, description="Billing renewal frequency")
    payment_method: str = Field(default="Credit Card", description="Payment method used")
    auto_renew: int = Field(default=1, ge=0, le=1, description="Whether auto-renewal is active (1 = True, 0 = False)")
    usage_frequency_per_month: int = Field(..., ge=0, le=31, description="Number of active days in the past 30 days")
    hours_used_per_month: float = Field(..., ge=0.0, description="Total hours consumed in the past 30 days")
    days_since_last_login: int = Field(..., ge=0, description="Inactivity recency: days elapsed since last session")
    support_tickets_last_3m: int = Field(default=0, ge=0, description="Support tickets created in the last 3 months")
    number_of_profiles_or_devices: int = Field(default=1, ge=1, le=20, description="Connected profiles or active devices")
    subscription_age_months: int = Field(default=6, ge=1, description="Tenure of the subscription account in months")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "subscription_name": "Adobe Creative Cloud",
                "category": "Productivity",
                "plan_tier": "Premium",
                "monthly_fee": 54.99,
                "billing_cycle": "Monthly",
                "payment_method": "Credit Card",
                "auto_renew": 1,
                "usage_frequency_per_month": 1,
                "hours_used_per_month": 0.5,
                "days_since_last_login": 45,
                "support_tickets_last_3m": 0,
                "number_of_profiles_or_devices": 1,
                "subscription_age_months": 14
            }
        }
    )

class EngineeredMetrics(BaseModel):
    """Derived economic metrics computed on the subscription."""
    cost_per_active_hour: float = Field(..., description="Cost per active consumption hour in USD")
    inactivity_ratio: float = Field(..., description="Ratio of inactivity to a 30-day billing cycle")
    annual_projected_cost: float = Field(..., description="Annualized cost in USD")

class SubscriptionPredictionOutput(BaseModel):
    """Output prediction for an evaluated subscription."""
    subscription_name: str
    monthly_fee: float
    is_wasteful: int = Field(..., description="1 = Underutilized/Wasteful, 0 = Well Utilized")
    waste_probability: float = Field(..., description="Probability of wastefulness (0.0% to 100.0%)")
    risk_level: str = Field(..., description="Risk categorization: LOW, MODERATE, or HIGH RISK")
    recommendation: str = Field(..., description="Actionable financial recommendation")
    engineered_metrics: EngineeredMetrics

class BatchPredictionRequest(BaseModel):
    """Batch evaluation payload containing multiple subscriptions."""
    subscriptions: List[SubscriptionInput] = Field(..., min_length=1, description="List of subscription records to audit")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "subscriptions": [
                    {
                        "subscription_name": "Adobe Creative Cloud",
                        "category": "Productivity",
                        "plan_tier": "Premium",
                        "monthly_fee": 54.99,
                        "billing_cycle": "Monthly",
                        "payment_method": "Credit Card",
                        "auto_renew": 1,
                        "usage_frequency_per_month": 0,
                        "hours_used_per_month": 0.0,
                        "days_since_last_login": 50,
                        "support_tickets_last_3m": 0,
                        "number_of_profiles_or_devices": 1,
                        "subscription_age_months": 12
                    },
                    {
                        "subscription_name": "Spotify",
                        "category": "Streaming",
                        "plan_tier": "Standard",
                        "monthly_fee": 10.99,
                        "billing_cycle": "Monthly",
                        "payment_method": "Debit Card",
                        "auto_renew": 1,
                        "usage_frequency_per_month": 26,
                        "hours_used_per_month": 40.0,
                        "days_since_last_login": 1,
                        "support_tickets_last_3m": 0,
                        "number_of_profiles_or_devices": 2,
                        "subscription_age_months": 24
                    }
                ]
            }
        }
    )

class BatchPredictionResponse(BaseModel):
    """Summary of batch audit and financial waste analysis."""
    total_subscriptions: int
    wasteful_subscriptions_count: int
    waste_rate_percentage: float
    total_monthly_spend: float
    total_potential_monthly_savings: float
    total_potential_annual_savings: float
    results: List[SubscriptionPredictionOutput]

class HealthResponse(BaseModel):
    """API health status and model availability."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str

class ModelMetadataResponse(BaseModel):
    """Trained model hyperparameters and performance metrics."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_type: str
    pipeline_steps: List[str]
    hyperparameters: Dict[str, Any]
    features: Dict[str, List[str]]
    evaluation_metrics: Dict[str, float]
    saved_at: str
