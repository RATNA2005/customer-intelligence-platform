import os
import sys

# Ensure root directory is in sys.path to resolve src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.models_registry import LocalModelRegistry


app = FastAPI(
    title="Customer Intelligence Platform API",
    description="Production-grade API serving Churn, CLV, Segmentation, and Recommendations",
    version="1.0.0"
)

# Initialize registry and load models
registry = LocalModelRegistry()
PROCESSED_DIR = "data/processed"

def get_customer_features(customer_id: str) -> Dict[str, Any]:
    features_path = os.path.join(PROCESSED_DIR, "inference_features.parquet")
    if not os.path.exists(features_path):
        raise HTTPException(status_code=500, detail="Feature store not found. Run pipelines first.")
        
    df_feat = pd.read_parquet(features_path)
    cust_row = df_feat[df_feat["customer_id"] == customer_id]
    if cust_row.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found.")
        
    return cust_row.iloc[0].to_dict()

# --- Schemas ---
class PredictionRequest(BaseModel):
    customer_id: str

class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    is_churn_risk: bool
    explanation: Dict[str, float]

class CLVResponse(BaseModel):
    customer_id: str
    predicted_12month_clv: float

class SegmentResponse(BaseModel):
    customer_id: str
    segment_id: int
    segment_name: str
    metrics: Dict[str, float]

class RecommendationResponse(BaseModel):
    customer_id: str
    recommended_product_ids: List[str]
    is_cold_start: bool

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    loaded_models = []
    for model_name in ["churn_model", "clv_model", "segmentation_model", "recommender_model"]:
        try:
            registry.load_model(model_name)
            loaded_models.append(model_name)
        except Exception:
            pass
    return {
        "status": "healthy" if len(loaded_models) == 4 else "degraded",
        "models_loaded": loaded_models
    }

@app.post("/predict/churn", response_model=ChurnResponse)
def predict_churn(request: PredictionRequest):
    try:
        model, metadata = registry.load_model("churn_model")
        feat_dict = get_customer_features(request.customer_id)
        
        # Filter inputs to model features
        X_df = pd.DataFrame([feat_dict])[metadata["features"]]
        
        prob = float(model.predict_proba(X_df)[0, 1])
        is_risk = prob >= 0.5
        
        # Calculate a simple local explanation: feature values scaled by feature importance
        importances = model.feature_importances_
        explanation = {}
        for f, imp in zip(metadata["features"], importances):
            val = float(X_df[f].iloc[0])
            # Focus explanation on Recency, Frequency, Monetary, Spending ratio
            if f in ["recency", "frequency", "monetary", "spending_ratio", "tenure", "age"]:
                explanation[f] = round(val * imp, 4)
                
        # Sort features by impact on explanation
        explanation = dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:4])
        
        return {
            "customer_id": request.customer_id,
            "churn_probability": prob,
            "is_churn_risk": is_risk,
            "explanation": explanation
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/predict/clv", response_model=CLVResponse)
def predict_clv(request: PredictionRequest):
    try:
        model, metadata = registry.load_model("clv_model")
        feat_dict = get_customer_features(request.customer_id)
        
        X_df = pd.DataFrame([feat_dict])[metadata["features"]]
        pred_clv = float(model.predict(X_df)[0])
        pred_clv = max(0.0, pred_clv) # ensure non-negative CLV
        
        return {
            "customer_id": request.customer_id,
            "predicted_12month_clv": round(pred_clv, 2)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CLV prediction failed: {str(e)}")

@app.post("/segment", response_model=SegmentResponse)
def predict_segment(request: PredictionRequest):
    try:
        seg_pipeline, metadata = registry.load_model("segmentation_model")
        feat_dict = get_customer_features(request.customer_id)
        
        scaler = seg_pipeline["scaler"]
        kmeans = seg_pipeline["model"]
        
        X_raw = pd.DataFrame([feat_dict])[metadata["features"]]
        X_scaled = scaler.transform(X_raw)
        
        segment_id = int(kmeans.predict(X_scaled)[0])
        
        # Business mapping of cluster IDs to segment profiles based on cluster centers
        # We define consistent profiles:
        # Segment 0: Hibernating / Churn Risk
        # Segment 1: High-Value VIP
        # Segment 2: Loyal / Active
        # Segment 3: New / Low Engagement
        segment_names = {
            0: "Hibernating (At Churn Risk)",
            1: "High-Value VIPs",
            2: "Active Loyalists",
            3: "Recent Sign-ups (Low Activity)"
        }
        
        segment_name = segment_names.get(segment_id, f"Segment Cohort {segment_id}")
        
        metrics = {
            "recency": float(feat_dict["recency"]),
            "frequency": float(feat_dict["frequency"]),
            "monetary": float(feat_dict["monetary"])
        }
        
        return {
            "customer_id": request.customer_id,
            "segment_id": segment_id,
            "segment_name": segment_name,
            "metrics": metrics
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: PredictionRequest):
    try:
        recommender, _ = registry.load_model("recommender_model")
        
        # Load user transactions history
        transactions_path = os.path.join(PROCESSED_DIR, "transactions.parquet")
        if not os.path.exists(transactions_path):
            raise HTTPException(status_code=500, detail="Transactions history missing.")
            
        df_tx = pd.read_parquet(transactions_path)
        user_history = df_tx[df_tx["customer_id"] == request.customer_id]["product_id"].unique().tolist()
        
        is_cold = len(user_history) == 0
        recs = recommender.recommend(request.customer_id, user_history, k=5)
        
        return {
            "customer_id": request.customer_id,
            "recommended_product_ids": recs,
            "is_cold_start": is_cold
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
