import os
import sys

# Ensure root directory is in sys.path to resolve src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from src.models_registry import LocalModelRegistry


# Setup Page Configuration
st.set_page_config(
    page_title="AI-Powered Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

st.title("🎯 Customer Intelligence Platform")
st.markdown("### Production-Grade Predictive Analytics Dashboard")

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
PROCESSED_DIR = "data/processed"
registry = LocalModelRegistry()

# Initialize data helper
@st.cache_data
def load_all_customer_ids():
    features_path = os.path.join(PROCESSED_DIR, "inference_features.parquet")
    if os.path.exists(features_path):
        df = pd.read_parquet(features_path)
        return df["customer_id"].tolist()
    return []

@st.cache_data
def get_aggregate_stats():
    features_path = os.path.join(PROCESSED_DIR, "inference_features.parquet")
    if os.path.exists(features_path):
        df = pd.read_parquet(features_path)
        return {
            "total_customers": len(df),
            "avg_age": int(df["age"].mean()),
            "avg_tenure": int(df["tenure"].mean()),
            "avg_spending": float(df["monetary"].mean())
        }
    return None

# Sidebar controls
st.sidebar.header("Pipeline Operations")
st.sidebar.markdown("Execute end-to-end tasks on dataset updates:")

# Retrain trigger inside sidebar
if st.sidebar.button("Retrain All Models"):
    with st.spinner("Retraining model pipelines..."):
        try:
            from src.train import run_training_pipeline
            run_training_pipeline()
            st.sidebar.success("✓ Retrained and registered all models successfully!")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Error during training: {e}")

# Check API health
api_healthy = False
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    if response.status_code == 200:
        api_healthy = True
except Exception:
    pass

if api_healthy:
    st.sidebar.success(f"● API Connection Active ({API_URL})")
else:
    st.sidebar.warning("○ Running in Local Fallback Mode (FastAPI Offline)")

# Load data summaries
customer_ids = load_all_customer_ids()
stats = get_aggregate_stats()

if not customer_ids:
    st.warning("No processed customer data found. Please trigger pipeline execution first.")
else:
    # 1. Executive Summary Panel
    st.markdown("#### 📈 Business Performance Cohorts")
    col1, col2, col3, col4 = st.columns(4)
    if stats:
        col1.metric("Active Customer Base", f"{stats['total_customers']:,}")
        col2.metric("Average Customer Age", f"{stats['avg_age']} Years")
        col3.metric("Average Tenure", f"{stats['avg_tenure']} Days")
        col4.metric("Avg Life Spend Value", f"${stats['avg_spending']:.2f}")

    st.markdown("---")

    # 2. Individual Customer profiling section
    st.markdown("#### 🔍 Single Customer Intelligence profiling")
    selected_customer = st.selectbox("Select Customer ID", customer_ids)
    
    if selected_customer:
        c1, c2 = st.columns(2)
        
        # Load local info
        features_path = os.path.join(PROCESSED_DIR, "inference_features.parquet")
        df_feat = pd.read_parquet(features_path)
        cust_row = df_feat[df_feat["customer_id"] == selected_customer].iloc[0]
        
        # Call models (API with fallback)
        churn_prob = 0.0
        clv_pred = 0.0
        segment_name = "Unknown"
        recs = []
        explanation = {}
        
        if api_healthy:
            try:
                # Churn request
                churn_res = requests.post(f"{API_URL}/predict/churn", json={"customer_id": selected_customer}).json()
                churn_prob = churn_res["churn_probability"]
                explanation = churn_res["explanation"]
                
                # CLV request
                clv_res = requests.post(f"{API_URL}/predict/clv", json={"customer_id": selected_customer}).json()
                clv_pred = clv_res["predicted_12month_clv"]
                
                # Segment request
                seg_res = requests.post(f"{API_URL}/segment", json={"customer_id": selected_customer}).json()
                segment_name = seg_res["segment_name"]
                
                # Rec request
                rec_res = requests.post(f"{API_URL}/recommend", json={"customer_id": selected_customer}).json()
                recs = rec_res["recommended_product_ids"]
            except Exception as e:
                st.error(f"API scoring failed: {e}. Falling back to registry load.")
                api_healthy = False
                
        # Fallback local loading
        if not api_healthy:
            try:
                # Churn Fallback
                churn_model, ch_meta = registry.load_model("churn_model")
                X_df = pd.DataFrame([cust_row.to_dict()])[ch_meta["features"]]
                churn_prob = float(churn_model.predict_proba(X_df)[0, 1])
                explanation = {f: float(X_df[f].iloc[0] * imp) for f, imp in zip(ch_meta["features"], churn_model.feature_importances_) if f in ["recency", "frequency", "monetary"]}
                
                # CLV Fallback
                clv_model, clv_meta = registry.load_model("clv_model")
                clv_pred = float(clv_model.predict(X_df)[0])
                clv_pred = max(0.0, clv_pred)
                
                # Segment Fallback
                seg_pipeline, seg_meta = registry.load_model("segmentation_model")
                scaler = seg_pipeline["scaler"]
                kmeans = seg_pipeline["model"]
                X_seg = pd.DataFrame([cust_row.to_dict()])[seg_meta["features"]]
                X_scaled = scaler.transform(X_seg)
                seg_id = int(kmeans.predict(X_scaled)[0])
                segment_names = {
                    0: "Hibernating (At Churn Risk)",
                    1: "High-Value VIPs",
                    2: "Active Loyalists",
                    3: "Recent Sign-ups (Low Activity)"
                }
                segment_name = segment_names.get(seg_id, f"Segment Cohort {seg_id}")
                
                # Rec Fallback
                recommender, _ = registry.load_model("recommender_model")
                df_tx = pd.read_parquet(os.path.join(PROCESSED_DIR, "transactions.parquet"))
                user_history = df_tx[df_tx["customer_id"] == selected_customer]["product_id"].unique().tolist()
                recs = recommender.recommend(selected_customer, user_history, k=5)
            except Exception as le:
                st.error(f"Registry load failed: {le}")

        # Visual layout for selected customer
        with c1:
            st.markdown("##### 📌 Customer Bio-Demographics")
            st.write(f"**Customer ID:** `{selected_customer}`")
            st.write(f"**Age:** {cust_row['age']} Years")
            st.write(f"**Country:** {cust_row['customer_id'][:1]}") # Parse country from encoding or metadata if needed
            st.write(f"**Tenure on Platform:** {int(cust_row['tenure'])} Days")
            st.write(f"**Historical Total Spend:** ${cust_row['monetary']:.2f}")
            st.write(f"**Historical Purchases:** {int(cust_row['frequency'])} Orders")
            
            st.markdown("##### 🏷️ Cohort Classification")
            st.info(f"**Predicted Segment Profile:** {segment_name}")
            
        with c2:
            st.markdown("##### 🔮 Predictive Health Checks")
            
            # Churn gauge display
            st.write("**Churn Probability Estimate**")
            churn_pct = int(churn_prob * 100)
            if churn_pct < 30:
                st.success(f"🟩 Low Risk: {churn_pct}%")
            elif churn_pct < 60:
                st.warning(f"🟨 Moderate Risk: {churn_pct}%")
            else:
                st.error(f"🟥 High Churn Risk: {churn_pct}%")
            
            # CLV prediction gauge
            st.write("**Predicted 12-Month Future Value**")
            st.metric("Expected Revenue Contribution", f"${clv_pred:.2f}")

            # Recommender Panel
            st.markdown("##### 🛒 Recommended Products for User")
            if recs:
                recs_df = pd.DataFrame({"Recommended Product ID": recs})
                st.dataframe(recs_df, use_container_width=True)
            else:
                st.write("No recommendations generated.")

        # Explanations view
        if explanation:
            st.markdown("##### 💡 Explainable AI (Local Feature Attribution)")
            st.write("The following metrics have the strongest influence on the customer's churn risk evaluation:")
            
            fig, ax = plt.subplots(figsize=(8, 3))
            feats = list(explanation.keys())
            scores = list(explanation.values())
            
            sns.barplot(x=scores, y=feats, ax=ax, palette="RdYlGn_r")
            ax.set_xlabel("Relative Attribution Weight")
            ax.set_ylabel("Customer Feature")
            st.pyplot(fig)
