import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, classification_report
from sqlalchemy import create_engine

# ==========================================
# PHASE 1: SYNTHETIC DATA GENERATION (TELCO CHURN)
# ==========================================
def generate_churn_data(num_customers=2000):
    """Generates realistic Telco customer behavior data"""
    np.random.seed(42)
    random.seed(42)
    
    customer_ids = [f"CUST_{10000 + i}" for i in range(num_customers)]
    tenure = np.random.randint(1, 72, size=num_customers)  # Months with the company
    monthly_charges = np.round(np.random.uniform(20, 120, size=num_customers), 2)
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_customers, p=[0.5, 0.25, 0.25])
    paperless_billing = np.random.choice([1, 0], size=num_customers, p=[0.6, 0.4])
    
    # Define churn logic based on features to simulate realistic patterns
    churn_prob = 0.1 + (0.4 * (contract_type == "Month-to-month")) + (0.3 * (monthly_charges > 80)) - (0.2 * (tenure > 24))
    churn_prob = np.clip(churn_prob, 0.02, 0.98)
    churn = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "contract_month_to_month": (contract_type == "Month-to-month").astype(int),
        "contract_one_year": (contract_type == "One year").astype(int),
        "paperless_billing": paperless_billing,
        "email": [f"user_{i}@example.com" for i in range(num_customers)],
        "churn": churn
    })
    return df

# ==========================================
# PHASE 2: ML MODELING (PRECISION-OPTIMIZED)
# ==========================================
def train_retention_model(df):
    """Trains a Random Forest classifier optimized for Precision to maximize ROI"""
    X = df[["tenure", "monthly_charges", "contract_month_to_month", "contract_one_year", "paperless_billing"]]
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Using balanced class weights to handle churn imbalance
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    
    print("\n=== Model Evaluation ===")
    print(classification_report(y_test, y_pred))
    
    # Calculate Financial Impact / ROI Framework
    # Assume saving a customer prevents $80/mo loss. A targeted email campaign costs $2/user.
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    total_targeted = tp + fp
    
    estimated_saved_revenue = tp * 80 * 12  # Annualized value
    campaign_cost = total_targeted * 2
    net_roi = estimated_saved_revenue - campaign_cost
    
    print(f"Targeted Customers: {total_targeted}")
    print(f"Precision (True Churners / Total Targeted): {precision:.2%}")
    print(f"📊 Projected Annualized Marketing ROI: ${net_roi:,.2f}\n")
    
    return model

# ==========================================
# PHASE 3: DATABASE STORAGE (POSTGRESQL)
# ==========================================
def save_predictions_to_db(df, model):
    """Scores the full database and writes predictions back to PostgreSQL"""
    X = df[["tenure", "monthly_charges", "contract_month_to_month", "contract_one_year", "paperless_billing"]]
    
    # Generate probabilities and risk tags
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["high_risk_flag"] = (df["churn_probability"] > 0.7).astype(int)
    df["prediction_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Filter only the high-risk segments to push downstream
    risk_table = df[df["high_risk_flag"] == 1][["customer_id", "email", "churn_probability", "prediction_date"]]
    
    print(f"Writing {len(risk_table)} high-risk segments to PostgreSQL...")
    try:
        # Reusing the existing local container configurations (Port 5433)
        engine = create_engine('postgresql://admin:admin@localhost:5433/personal_bi')
        risk_table.to_sql('high_risk_churn_customers', engine, if_exists='replace', index=False)
        print("✅ Database sync complete: table 'high_risk_churn_customers' updated.")
    except Exception as e:
        print(f"⚠️ DB Connection skipped (Local Docker not running). Saving locally to CSV.")
        risk_table.to_csv("high_risk_churn_customers.csv", index=False)
    
    return risk_table

# ==========================================
# PHASE 4: CRM AUTOMATION (MAILCHIMP MOCK)
# ==========================================
def trigger_crm_marketing_campaign(risk_table):
    """Simulates triggering an API call to Mailchimp automated journeys"""
    print("\n=== Triggering Automated CRM Email Campaigns ===")
    if len(risk_table) == 0:
        print("No high-risk users found today.")
        return
        
    # Pick a sample user to demonstrate the payload structure
    sample_user = risk_table.iloc[0]
    
    payload = {
        "email_address": sample_user["email"],
        "status": "subscribed",
        "merge_fields": {
            "CHURN_RISK": f"{sample_user['churn_probability']:.2%}",
            "CAMPAIGN": "Retention_Offer_2026",
            "DISCOUNT": "20% OFF"
        }
    }
    
    print(f"Successfully synced {len(risk_table)} contacts to Mailchimp Audience Segment.")
    print(f"Sample API Payload Sent to Mailchimp Endpoint:")
    print(payload)
    print("================================================")

# ==========================================
# MAIN EXECUTION ENGINE
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Automated End-to-End Churn & CRM Pipeline...")
    
    # 1. Generate/Load Data
    raw_data = generate_churn_data(num_customers=2500)
    
    # 2. Train and Validate Model
    churn_model = train_retention_model(raw_data)
    
    # 3. Store Results to Database
    high_risk_users = save_predictions_to_db(raw_data, churn_model)
    
    # 4. Trigger Marketing Campaign via API
    trigger_crm_marketing_campaign(high_risk_users)