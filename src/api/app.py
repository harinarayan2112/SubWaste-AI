"""
app.py
------
FastAPI production application for SubWaste AI.

Key ML Engineering & API Concepts Demonstrated:
1. Lifespan Model Management:
   The machine learning pipeline is loaded once into memory during application
   startup rather than on every HTTP request, minimizing latency.
2. Comprehensive Endpoints:
   - GET /health: Healthcheck for Kubernetes/Docker/uptime monitors.
   - GET /model/metadata: Model audit trail with training metrics.
   - POST /predict: Single subscription audit.
   - POST /predict/batch: Multiple subscriptions with aggregated financial leakage analysis.
   - POST /predict/csv: Direct CSV file upload for batch subscription auditing.
3. CORS & Middleware:
   Preconfigured for cross-origin requests, preparing the backend for future UI integration.
4. Robust Error Handling:
   Clear HTTP status codes (200, 400, 422, 503) and diagnostic JSON error responses.
"""

import os
import sys
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles

from src.predict import SubWastePredictor
from src.api.schemas import (
    SubscriptionInput,
    SubscriptionPredictionOutput,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelMetadataResponse,
    EngineeredMetrics
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("subwaste-api")

# Global predictor instance
predictor: SubWastePredictor = None
model_metadata: Dict[str, Any] = {}

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "subwaste_pipeline.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes ML predictor once during startup and verifies model availability.
    """
    global predictor, model_metadata
    logger.info("Initializing SubWaste AI API...")
    
    # Check model artifact
    if os.path.exists(MODEL_PATH):
        try:
            predictor = SubWastePredictor(model_path=MODEL_PATH)
            logger.info("Machine learning pipeline successfully loaded into memory.")
        except Exception as e:
            logger.error(f"Failed to load model pipeline: {e}")
            predictor = None
    else:
        logger.warning(f"Model file not found at '{MODEL_PATH}'. Run training pipeline first.")
        predictor = None
        
    # Check metadata
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                model_metadata = json.load(f)
            logger.info("Model metadata loaded.")
        except Exception as e:
            logger.warning(f"Could not read metadata: {e}")
            model_metadata = {}
            
    yield
    logger.info("Shutting down SubWaste AI API.")

app = FastAPI(
    title="SubWaste AI — Subscription Waste Intelligence API",
    description=(
        "Production REST API for identifying underutilized and economically wasteful "
        "recurring subscriptions using machine learning and behavioral telemetry."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for future frontend / dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for Frontend Dashboard
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def get_active_predictor() -> SubWastePredictor:
    """Helper to ensure predictor is initialized or raise 503."""
    global predictor
    if predictor is None:
        if os.path.exists(MODEL_PATH):
            predictor = SubWastePredictor(model_path=MODEL_PATH)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model artifact not found. Please train the model using 'python src/train.py'."
            )
    return predictor

@app.get("/", tags=["Dashboard"], include_in_schema=False)
async def serve_dashboard():
    """Serves the interactive glassmorphic web dashboard UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "app": "SubWaste AI API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs"
    }

@app.get("/api", tags=["General"])
async def api_overview():
    """API overview endpoint providing system status and documentation links."""
    return {
        "app": "SubWaste AI API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "dashboard": "/",
        "description": "Predict subscription waste and underutilization using behavioral telemetry."
    }

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Health check endpoint for container orchestrators and uptime monitors."""
    is_loaded = predictor is not None or os.path.exists(MODEL_PATH)
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version="1.0.0-randomforest",
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Diagnostics"])
async def get_model_metadata():
    """Retrieve hyperparameters, feature sets, and held-out test evaluation metrics."""
    if not model_metadata and os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            data = json.load(f)
        return data
    elif model_metadata:
        return model_metadata
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata artifact not found."
        )

@app.post("/predict", response_model=SubscriptionPredictionOutput, tags=["Inference"])
async def predict_subscription(subscription: SubscriptionInput):
    """
    Evaluate a single subscription for underutilization and economic waste.
    """
    model = get_active_predictor()
    record = subscription.model_dump()
    
    # Handle Enums to string conversion
    for key, val in record.items():
        if hasattr(val, "value"):
            record[key] = val.value
            
    try:
        prediction = model.predict_single(record)
        return SubscriptionPredictionOutput(
            subscription_name=prediction["subscription_name"],
            monthly_fee=prediction["monthly_fee"],
            is_wasteful=prediction["is_wasteful"],
            waste_probability=prediction["waste_probability"],
            risk_level=prediction["risk_level"],
            recommendation=prediction["recommendation"],
            engineered_metrics=EngineeredMetrics(**prediction["engineered_metrics"])
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_subscriptions(payload: BatchPredictionRequest):
    """
    Audit a list of subscriptions, returning individual evaluations and aggregated financial waste savings.
    """
    model = get_active_predictor()
    results = []
    
    total_spend = 0.0
    total_wasteful = 0
    monthly_waste_leak = 0.0
    
    for sub in payload.subscriptions:
        record = sub.model_dump()
        for key, val in record.items():
            if hasattr(val, "value"):
                record[key] = val.value
                
        pred = model.predict_single(record)
        sub_out = SubscriptionPredictionOutput(
            subscription_name=pred["subscription_name"],
            monthly_fee=pred["monthly_fee"],
            is_wasteful=pred["is_wasteful"],
            waste_probability=pred["waste_probability"],
            risk_level=pred["risk_level"],
            recommendation=pred["recommendation"],
            engineered_metrics=EngineeredMetrics(**pred["engineered_metrics"])
        )
        results.append(sub_out)
        
        fee = pred["monthly_fee"]
        total_spend += fee
        if pred["is_wasteful"] == 1:
            total_wasteful += 1
            monthly_waste_leak += fee
            
    num_subs = len(payload.subscriptions)
    waste_rate = round((total_wasteful / num_subs) * 100, 1) if num_subs > 0 else 0.0
    annual_savings = round(monthly_waste_leak * 12.0, 2)
    
    return BatchPredictionResponse(
        total_subscriptions=num_subs,
        wasteful_subscriptions_count=total_wasteful,
        waste_rate_percentage=waste_rate,
        total_monthly_spend=round(total_spend, 2),
        total_potential_monthly_savings=round(monthly_waste_leak, 2),
        total_potential_annual_savings=annual_savings,
        results=results
    )

@app.post("/predict/csv", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Audit subscriptions directly from an uploaded CSV file.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a CSV (.csv) file."
        )
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {e}"
        )
        
    required_cols = [
        "subscription_name", "monthly_fee", "usage_frequency_per_month",
        "hours_used_per_month", "days_since_last_login"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV is missing required columns: {missing}"
        )
        
    # Provide defaults for optional columns if absent
    if "category" not in df.columns:
        df["category"] = "Streaming"
    if "plan_tier" not in df.columns:
        df["plan_tier"] = "Standard"
    if "billing_cycle" not in df.columns:
        df["billing_cycle"] = "Monthly"
    if "payment_method" not in df.columns:
        df["payment_method"] = "Credit Card"
    if "auto_renew" not in df.columns:
        df["auto_renew"] = 1
    if "support_tickets_last_3m" not in df.columns:
        df["support_tickets_last_3m"] = 0
    if "number_of_profiles_or_devices" not in df.columns:
        df["number_of_profiles_or_devices"] = 1
    if "subscription_age_months" not in df.columns:
        df["subscription_age_months"] = 6
        
    # Convert dataframe rows to SubscriptionInput objects
    sub_inputs = []
    for _, row in df.iterrows():
        try:
            item = SubscriptionInput(
                subscription_name=str(row["subscription_name"]),
                category=str(row["category"]),
                plan_tier=str(row["plan_tier"]),
                monthly_fee=float(row["monthly_fee"]),
                billing_cycle=str(row["billing_cycle"]),
                payment_method=str(row["payment_method"]),
                auto_renew=int(row["auto_renew"]),
                usage_frequency_per_month=int(row["usage_frequency_per_month"]),
                hours_used_per_month=float(row["hours_used_per_month"]),
                days_since_last_login=int(row["days_since_last_login"]),
                support_tickets_last_3m=int(row["support_tickets_last_3m"]),
                number_of_profiles_or_devices=int(row["number_of_profiles_or_devices"]),
                subscription_age_months=int(row["subscription_age_months"])
            )
            sub_inputs.append(item)
        except Exception as e:
            logger.warning(f"Skipping malformed row: {e}")
            
    if not sub_inputs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV contained no valid subscription rows."
        )
        
    # Run batch evaluation
    batch_req = BatchPredictionRequest(subscriptions=sub_inputs)
    return await predict_batch_subscriptions(batch_req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
