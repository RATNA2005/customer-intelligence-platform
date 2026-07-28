import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_mock_data(output_dir="data/raw", num_customers=1000, num_products=50, num_transactions=20000):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating mock data in '{output_dir}'...")

    # 1. Generate Products
    categories = ["Electronics", "Apparel", "Home & Kitchen", "Books", "Beauty"]
    product_names = {
        "Electronics": ["Smartphone", "Laptop", "Headphones", "Smartwatch", "Bluetooth Speaker", "Tablet", "Charger", "Camera", "Monitor", "Keyboard"],
        "Apparel": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Socks", "Sweater", "Dress", "Hat", "Scarfe", "Belt"],
        "Home & Kitchen": ["Blender", "Coffee Maker", "Toaster", "Air Fryer", "Vacuum Cleaner", "Dinner Set", "Knife Set", "Pan", "Mug", "Pillow"],
        "Books": ["Fiction Novel", "Sci-Fi Novel", "Biography", "Self-Help Book", "History Book", "Cookbook", "Mystery Thriller", "Poetry", "Comic Book", "Textbook"],
        "Beauty": ["Lipstick", "Moisturizer", "Sunscreen", "Perfume", "Shampoo", "Face Wash", "Mascara", "Foundation", "Hair Dryer", "Nail Polish"]
    }

    products = []
    for cat in categories:
        names = product_names[cat]
        for i, name in enumerate(names):
            p_id = f"PROD_{cat[:3].upper()}_{i:03d}"
            # Realistic price ranges per category
            if cat == "Electronics":
                price = round(random.uniform(50.0, 1200.0), 2)
            elif cat == "Apparel":
                price = round(random.uniform(15.0, 150.0), 2)
            elif cat == "Home & Kitchen":
                price = round(random.uniform(20.0, 300.0), 2)
            elif cat == "Books":
                price = round(random.uniform(8.0, 45.0), 2)
            else: # Beauty
                price = round(random.uniform(10.0, 100.0), 2)
            
            products.append({
                "product_id": p_id,
                "name": name,
                "category": cat,
                "price": price
            })
    
    df_products = pd.DataFrame(products)
    
    # Add a few corrupt products for testing validation
    corrupt_products = [
        {"product_id": "PROD_ERR_001", "name": "Negative Price Item", "category": "Electronics", "price": -10.0},
        {"product_id": "PROD_ERR_002", "name": "Missing Category Item", "category": None, "price": 25.0}
    ]
    df_products_raw = pd.concat([df_products, pd.DataFrame(corrupt_products)], ignore_index=True)
    df_products_raw.to_csv(os.path.join(output_dir, "products.csv"), index=False)
    print(f"Saved {len(df_products_raw)} products (with {len(corrupt_products)} corrupt).")

    # 2. Generate Customers
    countries = ["US", "CA", "UK", "DE", "FR"]
    country_weights = [0.5, 0.15, 0.15, 0.1, 0.1]
    
    # Latent customer profiles: High Value, Loyal, New, At Risk, Churned
    profiles = ["High_Value", "Loyal", "New", "At_Risk", "Churned"]
    profile_weights = [0.1, 0.3, 0.2, 0.2, 0.2]
    
    customers = []
    end_date = datetime(2026, 7, 25) # Current project date reference
    
    for i in range(1, num_customers + 1):
        c_id = f"CUST_{i:06d}"
        profile = random.choices(profiles, weights=profile_weights)[0]
        country = random.choices(countries, weights=country_weights)[0]
        age = random.randint(18, 75)
        
        # Tenure based on profile
        if profile == "New":
            tenure_days = random.randint(1, 45)
        elif profile in ["High_Value", "Loyal"]:
            tenure_days = random.randint(180, 730)
        else: # At Risk, Churned
            tenure_days = random.randint(200, 730)
            
        signup_date = end_date - timedelta(days=tenure_days)
        
        customers.append({
            "customer_id": c_id,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "age": age,
            "country": country,
            "latent_profile": profile  # Useful for generating transactions, removed in validation
        })
        
    df_customers = pd.DataFrame(customers)
    
    # Add a few corrupt customers
    corrupt_customers = [
        {"customer_id": "CUST_ERR_001", "signup_date": "2026-07-30", "age": 12, "country": "US", "latent_profile": "New"}, # Underage
        {"customer_id": "CUST_ERR_002", "signup_date": "2024-05-12", "age": 45, "country": "JP", "latent_profile": "Loyal"} # Invalid country code
    ]
    df_customers_raw = pd.concat([df_customers, pd.DataFrame(corrupt_customers)], ignore_index=True)
    # Drop latent_profile for final output but we use df_customers for transaction generation
    df_customers_raw.drop(columns=["latent_profile"], errors="ignore").to_csv(os.path.join(output_dir, "customers.csv"), index=False)
    print(f"Saved {len(df_customers_raw)} customers (with {len(corrupt_customers)} corrupt).")

    # 3. Generate Transactions
    transactions = []
    tx_id_counter = 1
    
    # Map category index to profiles for recommender bias
    category_preferences = {
        "High_Value": ["Electronics", "Apparel"],
        "Loyal": ["Home & Kitchen", "Books", "Beauty"],
        "New": ["Books", "Apparel"],
        "At_Risk": ["Electronics", "Beauty"],
        "Churned": ["Apparel", "Home & Kitchen"]
    }
    
    for _, cust in df_customers.iterrows():
        c_id = cust["customer_id"]
        profile = cust["latent_profile"]
        signup = datetime.strptime(cust["signup_date"], "%Y-%m-%d")
        
        # Determine number of transactions based on profile
        if profile == "High_Value":
            tx_count = random.randint(30, 80)
            spending_coef = 1.5
        elif profile == "Loyal":
            tx_count = random.randint(15, 45)
            spending_coef = 1.0
        elif profile == "New":
            tx_count = random.randint(1, 5)
            spending_coef = 0.8
        elif profile == "At_Risk":
            tx_count = random.randint(10, 30)
            spending_coef = 0.9
        else: # Churned
            tx_count = random.randint(5, 20)
            spending_coef = 0.7
            
        # Distribute transactions over time
        tx_dates = []
        if profile == "Churned":
            # Transactions stop at least 90 days ago
            last_possible_tx = end_date - timedelta(days=91)
            if last_possible_tx < signup:
                last_possible_tx = signup + timedelta(days=1)
            total_active_days = (last_possible_tx - signup).days
            for _ in range(tx_count):
                offset = random.randint(0, max(1, total_active_days))
                tx_dates.append(signup + timedelta(days=offset))
        elif profile == "At_Risk":
            # Active in early days, dry spell in the last 60 days
            last_possible_tx = end_date - timedelta(days=61)
            if last_possible_tx < signup:
                last_possible_tx = signup + timedelta(days=1)
            total_active_days = (last_possible_tx - signup).days
            for _ in range(tx_count):
                offset = random.randint(0, max(1, total_active_days))
                tx_dates.append(signup + timedelta(days=offset))
        else:
            # Active all the way up to end_date
            total_active_days = (end_date - signup).days
            for _ in range(tx_count):
                offset = random.randint(0, max(1, total_active_days))
                tx_dates.append(signup + timedelta(days=offset))
                
        tx_dates.sort()
        
        # Product selection bias based on category preferences
        preferred_cats = category_preferences[profile]
        pref_products = df_products[df_products["category"].isin(preferred_cats)]["product_id"].tolist()
        other_products = df_products[~df_products["category"].isin(preferred_cats)]["product_id"].tolist()
        
        for dt in tx_dates:
            # 80% preference bias
            if random.random() < 0.8 and pref_products:
                p_id = random.choice(pref_products)
            else:
                p_id = random.choice(other_products)
                
            prod_row = df_products[df_products["product_id"] == p_id].iloc[0]
            price = prod_row["price"]
            
            # High value profile purchases higher quantity occasionally
            if profile == "High_Value" and random.random() < 0.2:
                quantity = random.randint(2, 5)
            else:
                quantity = random.randint(1, 2)
                
            amount = round(price * quantity, 2)
            
            transactions.append({
                "transaction_id": f"TX_{tx_id_counter:08d}",
                "customer_id": c_id,
                "product_id": p_id,
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "quantity": quantity,
                "amount": amount
            })
            tx_id_counter += 1

    df_transactions = pd.DataFrame(transactions)
    
    # Add a few corrupt transactions
    corrupt_transactions = [
        {"transaction_id": "TX_ERR_00001", "customer_id": "CUST_000001", "product_id": "PROD_ERR_999", "timestamp": "2026-07-25 10:00:00", "quantity": 0, "amount": 0.0}, # Zero quantity and invalid product
        {"transaction_id": "TX_ERR_00002", "customer_id": "CUST_000001", "product_id": "PROD_ELE_000", "timestamp": "2026-07-25 11:00:00", "quantity": -5, "amount": -100.0} # Negative quantity
    ]
    df_tx_raw = pd.concat([df_transactions, pd.DataFrame(corrupt_transactions)], ignore_index=True)
    df_tx_raw.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    print(f"Saved {len(df_tx_raw)} transactions (with {len(corrupt_transactions)} corrupt).")
    print("Data generation complete!")

if __name__ == "__main__":
    generate_mock_data()
