import pytest
import pandas as pd
import pandera as pa
from src.schemas import CustomerSchema, ProductSchema, TransactionSchema
from src.ingest import validate_and_quarantine

def test_customer_schema_validation():
    # Valid customer dataframe
    valid_data = pd.DataFrame({
        "customer_id": ["CUST_000001", "CUST_000002"],
        "signup_date": ["2026-07-25 12:00:00", "2026-07-24 10:00:00"],
        "age": [30, 45],
        "country": ["US", "CA"]
    })
    
    # Coerced type check
    validated = CustomerSchema.validate(valid_data)
    assert validated.shape[0] == 2
    assert pd.api.types.is_datetime64_any_dtype(validated["signup_date"])

    # Invalid customer dataframe (underage and invalid country code)
    invalid_data = pd.DataFrame({
        "customer_id": ["CUST_ERR_001", "CUST_ERR_002"],
        "signup_date": ["2026-07-25 12:00:00", "2026-07-24 10:00:00"],
        "age": [12, 45],  # 12 is less than ge=18
        "country": ["US", "JP"] # JP is not in allowed list
    })
    
    with pytest.raises(pa.errors.SchemaErrors):
        CustomerSchema.validate(invalid_data, lazy=True)

def test_quarantine_splitting():
    mixed_data = pd.DataFrame({
        "customer_id": ["CUST_000001", "CUST_ERR_001"],
        "signup_date": ["2026-07-25 12:00:00", "2026-07-24 10:00:00"],
        "age": [30, 12],  # Second row invalid (underage)
        "country": ["US", "US"]
    })
    
    valid_df, invalid_df = validate_and_quarantine(mixed_data, CustomerSchema, "test_customers")
    
    assert len(valid_df) == 1
    assert len(invalid_df) == 1
    assert valid_df.iloc[0]["customer_id"] == "CUST_000001"
    assert invalid_df.iloc[0]["customer_id"] == "CUST_ERR_001"
    assert "validation_errors" in invalid_df.columns

def test_product_schema_validation():
    invalid_product = pd.DataFrame({
        "product_id": ["PROD_ERR_001"],
        "name": ["Negative Price Item"],
        "category": ["Electronics"],
        "price": [-10.0]  # price must be positive (> 0)
    })
    with pytest.raises(pa.errors.SchemaErrors):
        ProductSchema.validate(invalid_product, lazy=True)
