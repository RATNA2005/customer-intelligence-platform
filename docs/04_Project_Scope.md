# Project Scope, Boundaries, and System Assumptions

Defining project scope prevents scope creep and ensures the platform is optimized for our target scale and resources.

---

## 1. In-Scope Focus
To build a production-quality local environment that mirrors corporate deployment standards, the project includes:

* **Modular ML Pipeline:** Python-based pipelines with robust validation (using `Pandera`) to clean, engineer features, and train XGBoost/LightGBM/K-Means models.
* **Explainable AI (XAI):** Integrating global and local SHAP/LIME interpretations to ensure predictions are auditable by business teams.
* **Model Registry & Tracking:** Setting up MLflow to log parameters, metrics, run artifacts, and manage the model state transition lifecycle (Staging vs. Production).
* **API Layer:** FastAPI service with Pydantic validation, error handling, rate limiting, and standard request logging.
* **Business Interface:** A Streamlit dashboard displaying business KPIs, individual customer profiling sheets, and recommendations.
* **Testing & MLOps Infrastructure:** Local Docker files for reproducibility, unit testing via Pytest, and validation checks.

---

## 2. Out-of-Scope (Future Enhancements)
To maintain project focus, the following production configurations are excluded from the initial phase:

* **Real-time Streaming Ingestion:** Kafka/Flink components for sub-second streaming feature updates are excluded. All features are batch-updated.
* **Distributed Training:** We assume the dataset fits into memory on a single machine. Spark, Ray, or multinode CPU/GPU training clusters are not implemented.
* **SSO/OAuth2 Enterprise Authentication:** While APIs have basic security, deep enterprise authentication systems (like Okta/Active Directory integration) are out-of-scope.
* **High Availability Kubernetes Deployments:** Model serving uses standard Docker Compose rather than Kubernetes orchestration (K8s/Helm).

---

## 3. Core Assumptions
* **Data Refresh Rate:** Data updates occur overnight (batch style). A delay of up to 24 hours in updating user segments or churn likelihoods is acceptable.
* **Transaction Latency:** The PostgreSQL transactional store will handle transactional updates, while our analytical processing runs on isolated copies/exports to avoid locking production databases.
