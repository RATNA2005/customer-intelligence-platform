import pandera as pa
from pandera.typing import Series

# Define allowed categories and countries as constants
ALLOWED_COUNTRIES = ["US", "CA", "UK", "DE", "FR"]
ALLOWED_CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Books", "Beauty"]

class CustomerSchema(pa.DataFrameModel):
    customer_id: Series[str] = pa.Field(unique=True, coerce=True, nullable=False)
    signup_date: Series[pa.DateTime] = pa.Field(coerce=True, nullable=False)
    age: Series[int] = pa.Field(ge=18, le=100, coerce=True, nullable=False)
    country: Series[str] = pa.Field(isin=ALLOWED_COUNTRIES, coerce=True, nullable=False)

    class Config:
        strict = True
        coerce = True

class ProductSchema(pa.DataFrameModel):
    product_id: Series[str] = pa.Field(unique=True, coerce=True, nullable=False)
    name: Series[str] = pa.Field(coerce=True, nullable=False)
    category: Series[str] = pa.Field(isin=ALLOWED_CATEGORIES, coerce=True, nullable=False)
    price: Series[float] = pa.Field(gt=0, coerce=True, nullable=False)

    class Config:
        strict = True
        coerce = True

class TransactionSchema(pa.DataFrameModel):
    transaction_id: Series[str] = pa.Field(unique=True, coerce=True, nullable=False)
    customer_id: Series[str] = pa.Field(coerce=True, nullable=False)
    product_id: Series[str] = pa.Field(coerce=True, nullable=False)
    timestamp: Series[pa.DateTime] = pa.Field(coerce=True, nullable=False)
    quantity: Series[int] = pa.Field(ge=1, coerce=True, nullable=False)
    amount: Series[float] = pa.Field(ge=0, coerce=True, nullable=False)

    class Config:
        strict = True
        coerce = True
