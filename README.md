# 🔄 Cloud-Native CRM Automation & Churn Prediction

An end-to-end, production-grade machine learning pipeline that identifies high-risk telecommunications customers and automatically triggers targeted retention campaigns, translating predictive accuracy directly into business ROI.


## 🚀 Project Overview

Most churn prediction projects stop at model accuracy. This project bridges the gap between Data Science and Business Operations. It features a **Precision-optimized Random Forest classifier** designed to minimize wasteful marketing spend, calculates the projected financial impact, and acts on those predictions via an automated CRM pipeline.

### Core Architecture
1. **Synthetic Data Engine**: Generates realistic Telco customer profiles (tenure, billing, contract types).
2. **Precision-Optimized ML**: A Random Forest model with balanced class weights, tuned specifically to identify true churners without over-targeting.
3. **Data Warehousing**: High-risk customer segments (>70% churn probability) are automatically written back to a local **PostgreSQL** instance via SQLAlchemy.
4. **CRM Automation (API Mock)**: Generates automated JSON payloads tailored for **Mailchimp** automated journeys, delivering specific retention offers (e.g., 20% OFF) to the high-risk cohort.

## 💰 Business Impact (Sample Run)
Based on a simulated cohort of 2,000 customers:
- **Precision**: `54.41%` (True Churners / Total Targeted).
- **Actionable Insights**: Identified 569 high-risk segments.
- **Projected Annualized ROI**: **`$70,768.00`** (Calculated based on a $80/mo saved revenue minus a $2/user campaign cost).

## 🛠️ Tech Stack
- **Languages**: Python 3.13
- **Machine Learning**: Scikit-Learn (`RandomForestClassifier`, Classification Metrics)
- **Data Engineering**: Pandas, NumPy, SQLAlchemy, PostgreSQL
- **Deployment Strategy**: Designed to be containerized via Docker and orchestrated via AWS Lambda / EventBridge for daily batch runs.

## 🏁 Getting Started
1. Ensure your local PostgreSQL instance is running (Port 5433).
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn sqlalchemy psycopg2-binary
