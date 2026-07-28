FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies (required for XGBoost/LightGBM builds if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command (run the FastAPI server)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
