import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from src.models_registry import LocalModelRegistry
from src.recommender import ItemBasedRecommender

def train_churn_model(df_train, registry):
    print("\n--- Training Churn Model (XGBoost) ---")
    
    # Exclude targets, identifiers and has_transactions indicator
    feature_cols = [col for col in df_train.columns if col not in ["customer_id", "target_churn", "target_clv", "has_transactions"]]
    
    X = df_train[feature_cols]
    y = df_train["target_churn"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define XGBoost model
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
        "roc_auc": float(roc_auc_score(y_val, y_proba))
    }
    
    print(f"Validation Metrics: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
    
    # Register the model
    registry.register_model(
        model=model,
        name="churn_model",
        metrics=metrics,
        params=params,
        features=feature_cols
    )

def train_clv_model(df_train, registry):
    print("\n--- Training CLV Model (LightGBM) ---")
    
    feature_cols = [col for col in df_train.columns if col not in ["customer_id", "target_churn", "target_clv", "has_transactions"]]
    
    # We train CLV regression model on all active training customers
    X = df_train[feature_cols]
    y = df_train["target_clv"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    # Clip negative predictions to 0 as CLV must be non-negative
    y_pred = np.clip(y_pred, 0, None)
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    # Filter actual zero cases to avoid divide-by-zero, or add offset
    actual_non_zero = y_val > 0
    if actual_non_zero.sum() > 0:
        mape = np.mean(np.abs((y_val[actual_non_zero] - y_pred[actual_non_zero]) / y_val[actual_non_zero]))
    else:
        mape = 0.0
        
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape)
    }
    
    print(f"Validation Metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}, MAPE={metrics['mape']:.4f}")
    
    registry.register_model(
        model=model,
        name="clv_model",
        metrics=metrics,
        params=params,
        features=feature_cols
    )

def train_segmentation_model(df_inference, registry):
    print("\n--- Training Segmentation Model (KMeans) ---")
    
    # RFM clustering is computed on inference features (current state)
    cluster_features = ["recency", "frequency", "monetary"]
    X = df_inference[cluster_features].copy()
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans (we default to K=4 based on standard e-commerce RFM strategies)
    k = 4
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    
    labels = model.labels_
    score = silhouette_score(X_scaled, labels)
    
    metrics = {
        "silhouette_score": float(score),
        "n_clusters": k
    }
    print(f"Validation Metrics: Silhouette Score={metrics['silhouette_score']:.4f}")
    
    # We bundle the scaler and KMeans model together as a dictionary
    segmentation_pipeline = {
        "scaler": scaler,
        "model": model,
        "features": cluster_features
    }
    
    registry.register_model(
        model=segmentation_pipeline,
        name="segmentation_model",
        metrics=metrics,
        params={"n_clusters": k},
        features=cluster_features
    )

# ItemBasedRecommender class is imported from src.recommender

def train_recommender_model(processed_dir, registry):
    print("\n--- Training Recommender Model (Collaborative Cosine) ---")
    interactions_path = os.path.join(processed_dir, "recommender_matrix.parquet")
    interactions_df = pd.read_parquet(interactions_path)
    
    recommender = ItemBasedRecommender()
    recommender.fit(interactions_df)
    
    # Log random baseline metric: Average coverage of recommendations
    all_products = interactions_df["product_id"].unique()
    sample_cust = interactions_df["customer_id"].unique()[:10]
    
    recommendation_coverage = []
    for c_id in sample_cust:
        cust_bought = interactions_df[interactions_df["customer_id"] == c_id]["product_id"].tolist()
        recs = recommender.recommend(c_id, cust_bought, k=5)
        recommendation_coverage.extend(recs)
        
    unique_covered = len(set(recommendation_coverage))
    metrics = {
        "unique_products_recommended_sample": unique_covered,
        "sample_coverage_ratio": float(unique_covered / len(all_products))
    }
    print(f"Validation Metrics: Unique products recommended in sample (out of {len(all_products)}) = {unique_covered}")
    
    registry.register_model(
        model=recommender,
        name="recommender_model",
        metrics=metrics,
        params={"top_n_similar": 10},
        features=["product_id", "purchase_count"]
    )

def run_training_pipeline(processed_dir="data/processed"):
    registry = LocalModelRegistry()
    
    # Load feature sets
    df_train = pd.read_parquet(os.path.join(processed_dir, "train_features.parquet"))
    df_inference = pd.read_parquet(os.path.join(processed_dir, "inference_features.parquet"))
    
    train_churn_model(df_train, registry)
    train_clv_model(df_train, registry)
    train_segmentation_model(df_inference, registry)
    train_recommender_model(processed_dir, registry)
    print("\n[OK] Training pipeline completed successfully.")

if __name__ == "__main__":
    run_training_pipeline()
