import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_processed_data(processed_dir="data/processed"):
    customers = pd.read_parquet(os.path.join(processed_dir, "customers.parquet"))
    products = pd.read_parquet(os.path.join(processed_dir, "products.parquet"))
    transactions = pd.read_parquet(os.path.join(processed_dir, "transactions.parquet"))
    return customers, products, transactions

def build_features(reference_date_str, mode="train", processed_dir="data/processed"):
    """
    Build features for churn, CLV, and segmentation.
    If mode == "train", we split the dataset at the reference_date:
      - Features are calculated from transactions <= reference_date
      - Churn & CLV targets are calculated from transactions > reference_date (up to reference_date + 90 days)
    If mode == "inference", we calculate features up to reference_date (usually the current date)
      - Targets are not calculated (unknown)
    """
    customers, products, transactions = load_processed_data(processed_dir)
    
    ref_date = pd.to_datetime(reference_date_str)
    
    # 1. Filter transactions for features (history up to ref_date)
    history_tx = transactions[transactions["timestamp"] <= ref_date].copy()
    
    # Map products to transactions to get categories and prices
    history_tx = history_tx.merge(products, on="product_id", how="left")
    
    # 2. Compute RFM and behavioral features per customer
    customer_features = []
    
    for _, cust in customers.iterrows():
        cust_id = cust["customer_id"]
        signup_date = pd.to_datetime(cust["signup_date"])
        
        # Tenure in days relative to reference date
        tenure = (ref_date - signup_date).days
        if tenure <= 0:
            # Customer signed up after the reference date (exclude in train mode)
            if mode == "train":
                continue
            tenure = 1 # avoid division by zero in inference if signup matches ref_date
            
        # Get customer transactions in history
        cust_tx = history_tx[history_tx["customer_id"] == cust_id]
        
        # Standard RFM calculation
        if len(cust_tx) > 0:
            last_purchase = cust_tx["timestamp"].max()
            recency = (ref_date - last_purchase).days
            frequency = cust_tx["transaction_id"].nunique()
            monetary = cust_tx["amount"].sum()
            
            # Additional features
            avg_order_value = monetary / frequency if frequency > 0 else 0
            avg_quantity = cust_tx["quantity"].mean()
            
            # Category-wise purchase distribution
            cat_counts = cust_tx["category"].value_counts().to_dict()
            electronics_pct = cat_counts.get("Electronics", 0) / len(cust_tx)
            apparel_pct = cat_counts.get("Apparel", 0) / len(cust_tx)
            home_pct = cat_counts.get("Home & Kitchen", 0) / len(cust_tx)
            books_pct = cat_counts.get("Books", 0) / len(cust_tx)
            beauty_pct = cat_counts.get("Beauty", 0) / len(cust_tx)
            
            # Trend features (Last 45 days vs 45-90 days ago)
            recent_tx = cust_tx[cust_tx["timestamp"] > (ref_date - timedelta(days=45))]
            older_tx = cust_tx[(cust_tx["timestamp"] <= (ref_date - timedelta(days=45))) & 
                               (cust_tx["timestamp"] > (ref_date - timedelta(days=90)))]
            
            recent_monetary = recent_tx["amount"].sum()
            older_monetary = older_tx["amount"].sum()
            
            # Ratio representing spending acceleration or slowdown
            spending_ratio = recent_monetary / (older_monetary + 1.0) # Laplace smoothing
            
            has_transactions = 1
        else:
            recency = tenure
            frequency = 0
            monetary = 0.0
            avg_order_value = 0.0
            avg_quantity = 0.0
            electronics_pct = 0.0
            apparel_pct = 0.0
            home_pct = 0.0
            books_pct = 0.0
            beauty_pct = 0.0
            spending_ratio = 0.0
            has_transactions = 0
            
        # One-hot encode country
        countries = ["US", "CA", "UK", "DE", "FR"]
        country_dict = {f"country_{c}": 1 if cust["country"] == c else 0 for c in countries}
        
        feat_row = {
            "customer_id": cust_id,
            "age": cust["age"],
            "tenure": tenure,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_order_value": avg_order_value,
            "avg_quantity": avg_quantity,
            "electronics_pct": electronics_pct,
            "apparel_pct": apparel_pct,
            "home_pct": home_pct,
            "books_pct": books_pct,
            "beauty_pct": beauty_pct,
            "spending_ratio": spending_ratio,
            "has_transactions": has_transactions,
            **country_dict
        }
        
        # 3. Target calculation (if train mode)
        if mode == "train":
            forward_window_start = ref_date
            forward_window_end = ref_date + timedelta(days=90)
            
            # Future transactions for targets
            future_tx = transactions[(transactions["customer_id"] == cust_id) & 
                                     (transactions["timestamp"] > forward_window_start) & 
                                     (transactions["timestamp"] <= forward_window_end)]
            
            # Churn target: 1 if user made NO transaction in the next 90 days
            churn_target = 1 if len(future_tx) == 0 else 0
            # CLV target: sum of transaction amounts in the next 90 days
            clv_target = future_tx["amount"].sum()
            
            feat_row["target_churn"] = churn_target
            feat_row["target_clv"] = clv_target
            
        customer_features.append(feat_row)
        
    df_features = pd.DataFrame(customer_features)
    
    # Save features
    out_path = os.path.join(processed_dir, f"{mode}_features.parquet")
    df_features.to_parquet(out_path, index=False)
    print(f"[OK] Created {mode} features with shape {df_features.shape} saved to '{out_path}'")
    return df_features

def build_recommender_matrix(processed_dir="data/processed"):
    """
    Build user-product purchase frequency matrix for recommendation models.
    """
    _, _, transactions = load_processed_data(processed_dir)
    
    # Compute interaction count per customer-product pair
    user_item_interactions = transactions.groupby(["customer_id", "product_id"]).size().reset_index(name="purchase_count")
    
    out_path = os.path.join(processed_dir, "recommender_matrix.parquet")
    user_item_interactions.to_parquet(out_path, index=False)
    print(f"[OK] Created user-item interaction matrix with shape {user_item_interactions.shape} saved to '{out_path}'")
    return user_item_interactions

def run_feature_pipeline():
    # Training splits: reference date at 2026-04-25 (giving 90 days to 2026-07-25 for targets)
    build_features(reference_date_str="2026-04-25 00:00:00", mode="train")
    # Production scoring splits: reference date at 2026-07-25 (scoring active users using all data)
    build_features(reference_date_str="2026-07-25 00:00:00", mode="inference")
    
    build_recommender_matrix()

if __name__ == "__main__":
    run_feature_pipeline()
