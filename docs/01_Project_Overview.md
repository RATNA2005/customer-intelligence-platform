# AI-Powered Customer Intelligence Platform: Project Overview

The **AI-Powered Customer Intelligence Platform** is an enterprise-grade, end-to-end Machine Learning system designed to analyze customer behavior, predict key action events, and optimize customer lifetime value (CLV) for an online retail e-commerce platform.

By unifying customer interaction logs, transactional history, and demographic profiles, the platform provides businesses with actionable insights, personalized product recommendations, and early warnings for churn risk.

---

## 1. High-Level System Architecture

The platform is designed around modular, decoupled components to ensure scalability, ease of model maintenance, and reliable serving.

```mermaid
graph TB
    %% Data Source Layer
    subgraph Data Layer
        DB[(PostgreSQL Database)] --> FeatureStore[(Local Feature Store / Parquet)]
    end

    %% Pipeline Layer
    subgraph MLOps & Orchestration
        Pipeline[Data Ingestion & Feature Engineering Pipeline] --> MLflow[(MLflow Model Registry)]
    end
    FeatureStore --> Pipeline

    %% Model Serving Layer
    subgraph Model Serving Layer (FastAPI REST API)
        MLflow --> ChurnModel[Churn Model XGBoost]
        MLflow --> CLVModel[CLV Model LightGBM]
        MLflow --> SegModel[Segmentation KMeans/DBSCAN]
        MLflow --> RecModel[Recommender Collaborative/Content]
    end

    %% Frontend & Analytics
    subgraph Client Applications
        API[FastAPI Gateway] --> Dashboard[Streamlit Analytics Dashboard]
        API --> MarketingClient[Marketing Automation Integration]
    end

    ChurnModel --> API
    CLVModel --> API
    SegModel --> API
    RecModel --> API

    %% Monitoring
    subgraph Monitoring
        API --> Prometheus[Data Drift & Model Monitoring]
    end
```

---

## 2. Core Modules

The platform is composed of four core ML modules integrated under a unified API gateway:

1. **Churn Predictor:** Evaluates the probability of a customer leaving the platform within a defined non-activity window (e.g., 90 days).
2. **Customer Segmenter:** Groups customers dynamically based on behavioral patterns (Recency, Frequency, Monetary metrics) to support personalized marketing campaigns.
3. **CLV Regressor:** Forecasts the cumulative net revenue a customer will bring over a designated forward-looking period (e.g., 12 months).
4. **Recommendation Engine:** Generates personalized product suggestions for active users using a hybrid approach (collaborative filtering and item similarity).

---

## 3. Tech Stack

To reflect industry-standard practices, the project implements:
* **Storage:** PostgreSQL for transactional data; standardized Parquet files for feature engineering caches.
* **Model Pipeline:** Scikit-Learn, XGBoost, LightGBM, CatBoost, and Optuna for pipeline creation and hyperparameter tuning.
* **MLOps:** MLflow for tracking experiments and registering models.
* **Serving:** FastAPI for asynchronous REST APIs with Pydantic for request/response validation.
* **Frontend:** Streamlit for interactive business visualization.
* **CI/CD & Infrastructure:** Docker containerization, GitHub Actions, and unit testing via Pytest.
