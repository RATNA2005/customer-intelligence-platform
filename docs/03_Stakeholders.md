# Stakeholder Mapping and Model Consumption Patterns

An ML platform does not exist in isolation. It serves multiple stakeholders across the organization. Designing interfaces optimized for each team's needs ensures operational adoption.

---

## 1. Stakeholder Matrix

| Stakeholder | Business Goal | Interface Type | Consumption Frequency | Model Utilized |
| :--- | :--- | :--- | :--- | :--- |
| **Marketing Operations** | Target promotions, optimize ad campaigns, prevent churn. | CSV Export / Dashboard | Weekly Batch | Churn, Segmentation, Recommendations |
| **Finance & Executives** | Financial forecasting, budget planning (CAC limit checks). | Dashboard / Reports | Monthly | CLV |
| **Product Engineering** | Populate the "Recommended for You" shelf on checkout. | REST API | Real-Time (HTTP) | Recommendation Engine |
| **Customer Support** | Proactively reach out to VIP customers at risk of churn. | Dashboard / Alerts | Daily Batch | Churn, CLV |

---

## 2. Ingestion & Consumption Pipelines

### Batch Processing Pattern
* **Use Case:** Churn risk updates and segment reassessments do not need sub-second updates. 
* **Mechanism:** 
  1. A daily/weekly cron job triggers ingestion pipelines.
  2. The inference pipeline processes the entire user base database.
  3. Predictions are cached in PostgreSQL.
  4. Streamlit and internal BI tools pull these pre-computed results directly.

### Real-Time Serving Pattern
* **Use Case:** Checkout recommendations must adapt to the customer's current session basket.
* **Mechanism:**
  1. FastAPI handles incoming JSON payloads containing the user's ID and current context.
  2. The API queries active database states for feature retrieval.
  3. Pre-trained recommendation vectors are evaluated instantly.
  4. Top-K recommended product IDs are returned within a $< 200\text{ms}$ SLA window.
