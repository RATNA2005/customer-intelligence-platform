# Success Metrics and KPIs: Aligning ML with Business Value

For an ML project to be successful, standard statistical metrics must be translated into financial metrics. Below is the mapping framework our platform uses.

---

## 1. Metric Alignment Table

| System Module | Machine Learning Metric | Business KPI | Business Definition |
| :--- | :--- | :--- | :--- |
| **Churn Predictor** | **Recall:** $\ge 80\%$<br>**ROC-AUC:** $\ge 0.85$ | **Churn Rate Reduction** | Reduction in monthly active customer drop-offs. |
| **CLV Regressor** | **MAE:** $\le \$50$<br>**MAPE:** $\le 15\%$ | **CAC/LTV Ratio Optimization** | Maximizing marketing budget spend efficiency. |
| **Segmentation** | **Silhouette Score:** $\ge 0.45$ | **Campaign Conversion Uplift** | Conversion rate increase in targeted campaigns. |
| **Recommender** | **MAP@K (K=5):** $\ge 0.15$ | **Average Order Value (AOV) Uplift** | Increase in total cart value per purchase. |

---

## 2. Business Value Mathematical Formulas

### Churn Cost-Benefit Evaluation
Let:
* $C_{\text{coupon}}$ = Cost of a promotion coupon sent to a flagged customer (e.g., \$10)
* $V_{\text{customer}}$ = Expected value of a saved customer (e.g., \$150)
* $P_{\text{accept}}$ = Probability that a churner accepts the coupon and stays (e.g., $30\%$)

The net business value of our churn classification model is computed as:
$$\text{Net Value} = TP \times (P_{\text{accept}} \times V_{\text{customer}} - C_{\text{coupon}}) - FP \times C_{\text{coupon}}$$
Where:
* $TP$ (True Positives): Churners correctly identified and target-messaged.
* $FP$ (False Positives): Loyal customers incorrectly flagged who receive a coupon they didn't need (direct marketing waste).

This equation proves why **Precision** and **Recall** must be balanced. If Precision is too low ($FP$ is huge), we waste marketing budgets. If Recall is too low ($TP$ is small), we fail to save customers.

---

## 3. SLA & Operational Performance Metrics
To ensure the system works reliably under load:
* **API Response Latency:** 95th percentile response time ($p95$) for recommendations must be $\le 200\text{ms}$.
* **Model Inference Latency:** Batch prediction throughput of $10,000$ rows in $\le 30\text{seconds}$.
* **Data Validation Quality:** Ingestion pipelines must achieve $100\%$ schema adherence (rejected transactions quarantined automatically).
