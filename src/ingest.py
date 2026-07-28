import os
import pandas as pd
import pandera as pa
from src.schemas import CustomerSchema, ProductSchema, TransactionSchema

def validate_and_quarantine(df, schema_model, name):
    print(f"\nValidating {name}...")
    try:
        # First-pass lazy validation to detect errors
        validated_df = schema_model.validate(df, lazy=True)
        print(f"[OK] All {len(df)} rows in {name} passed validation.")
        return validated_df, pd.DataFrame(columns=df.columns)
    except pa.errors.SchemaErrors as err:
        # Extract row-level indices that failed validation
        failure_cases = err.failure_cases
        print(f"[WARNING] Validation failed for {name} with {len(failure_cases)} failures.")
        
        # Filter failures that have a valid row index in the original dataframe
        invalid_indices = []
        schema_failures = []
        for idx in failure_cases["index"].dropna().unique():
            invalid_indices.append(int(idx))
            
        # Get dataframe split
        if invalid_indices:
            invalid_df = df.loc[invalid_indices].copy()
            # Capture failure reasons in the quarantined file
            invalid_df["validation_errors"] = str(failure_cases.to_dict(orient="records"))
            valid_df = df.drop(index=invalid_indices).copy()
        else:
            invalid_df = df.copy()
            invalid_df["validation_errors"] = "Schema-level validation error (missing column or incorrect schema structure)"
            valid_df = pd.DataFrame(columns=df.columns)
            
        # Attempt to coerce and validate the cleaned dataframe
        if len(valid_df) > 0:
            try:
                valid_df = schema_model.validate(valid_df, lazy=True)
                print(f"[SUCCESS] Quarantined {len(invalid_df)} rows. {len(valid_df)} rows validated successfully.")
            except Exception as e:
                print(f"[ERROR] Secondary validation failed for {name}: {e}. Discarding whole batch.")
                invalid_df = df.copy()
                invalid_df["validation_errors"] = f"Failed second pass validation: {str(e)}"
                valid_df = pd.DataFrame(columns=df.columns)
        else:
            print(f"[ERROR] No valid rows remain after quarantine for {name}.")
            
        return valid_df, invalid_df

def run_ingestion(raw_dir="data/raw", processed_dir="data/processed", quarantine_dir="data/quarantine"):
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(quarantine_dir, exist_ok=True)
    
    datasets = [
        ("customers.csv", CustomerSchema, "customers"),
        ("products.csv", ProductSchema, "products"),
        ("transactions.csv", TransactionSchema, "transactions")
    ]
    
    for filename, schema, name in datasets:
        file_path = os.path.join(raw_dir, filename)
        if not os.path.exists(file_path):
            print(f"Error: {file_path} does not exist. Run data generation first.")
            continue
            
        df = pd.read_csv(file_path)
        valid_df, invalid_df = validate_and_quarantine(df, schema, name)
        
        # Save valid to parquet
        valid_path = os.path.join(processed_dir, f"{name}.parquet")
        valid_df.to_parquet(valid_path, index=False)
        print(f"Saved clean {name} to '{valid_path}'")
        
        # Save invalid to csv/quarantine
        if len(invalid_df) > 0:
            invalid_path = os.path.join(quarantine_dir, f"{name}_corrupt.csv")
            invalid_df.to_csv(invalid_path, index=False)
            print(f"Quarantined corrupt {name} to '{invalid_path}'")

if __name__ == "__main__":
    run_ingestion()
