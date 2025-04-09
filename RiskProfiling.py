# You’re giving the model the question paper — not the answer key

import streamlit as st
import pandas as pd
# import numpy as np
import sqlite3
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb

# Core Model Functions
def load_data():
    uploaded_db = st.file_uploader("Upload SQLite Database (.db)", type=["db"], key="sql_db")
    if uploaded_db is not None:
        try:
            temp_path = os.path.join("temp_uploaded.db")
            with open(temp_path, "wb") as f:
                f.write(uploaded_db.read())

            conn = sqlite3.connect(temp_path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            table_names = tables['name'].tolist()

            if not table_names:
                st.error("❌ No tables found in database.")
                conn.close()
                os.remove(temp_path)
                return None

            selected_table = st.selectbox("Select Table", table_names)
            df = pd.read_sql(f"SELECT * FROM {selected_table}", conn)
            conn.close()
            os.remove(temp_path)

            st.success(f"✅ Loaded table: {selected_table}")
            return df
        except Exception as e:
            st.error(f"❌ Error reading database: {e}")
    else:
        st.info("📥 Please upload a `.db` SQLite file.")
    return None

def suggest_features(df):
    recommended_features = [
        "credit_score", "income", "loan_amount", "employment_status",
        "age", "payment_history", "num_late_payments", "total_due",
        "total_paid", "account_age_months"
    ]
    missing = [feat for feat in recommended_features if feat not in df.columns]
    if missing:
        st.warning("⚠️ The following recommended features for better customer risk profiling are missing:")
        for m in missing:
            st.write(f"- {m}")
        st.info("💡 Consider enriching your dataset with these features to improve model performance.")
    else:
        st.success("✅ Your dataset includes all key features recommended for customer risk profiling.")

def preprocess(df):
    df.fillna(0, inplace=True)

    if "total_paid" in df.columns and "total_due" in df.columns:
        df["payment_ratio"] = df["total_paid"] / (df["total_due"] + 1)
    if "total_delay_days" in df.columns and "num_late_payments" in df.columns:
        df["avg_delay_days"] = df["total_delay_days"] / (df["num_late_payments"] + 1)

    target = "default" if "default" in df.columns else df.columns[-1]
    features = [col for col in df.columns if col not in ["customer_id", "name", target]]

    suggest_features(df)

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, features

def train_model(X, y):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    st.write("🔍 **Accuracy:**", round(acc, 3))
    st.write("📈 **AUC-ROC:**", round(auc, 3))
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))
    return y_proba, pd.qcut(y_proba, q=4, labels=["Low", "Medium", "High", "Critical"])

def predict_uploaded(model, scaler, features):
    uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"], key="predict_csv")
    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.write("📋 Prediction Data Preview", df_new.head())

        df_new.fillna(0, inplace=True)

        if "total_paid" in df_new.columns and "total_due" in df_new.columns:
            df_new["payment_ratio"] = df_new["total_paid"] / (df_new["total_due"] + 1)
        if "total_delay_days" in df_new.columns and "num_late_payments" in df_new.columns:
            df_new["avg_delay_days"] = df_new["total_delay_days"] / (df_new["num_late_payments"] + 1)

        df_new = df_new.drop(columns=["default"], errors="ignore")

        if not features:
            features = [col for col in df_new.columns if col not in ["customer_id", "name", "default"]]

        X_new = df_new[features]
        X_scaled = scaler.transform(X_new)

        predictions = model.predict_proba(X_scaled)[:, 1]
        df_new["default_probability"] = predictions
        df_new["risk_segment"] = pd.qcut(predictions, q=4, labels=["Low", "Medium", "High", "Critical"])

        st.write("📊 Prediction Results", df_new[["customer_id", "default_probability", "risk_segment"]])

def collect_feedback():
    st.sidebar.header("💬 Feedback")
    feedback = st.sidebar.text_area("Share your feedback or suggestions")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thanks for your feedback!")

def save_model(model, scaler):
    joblib.dump(model, "risk_model.joblib")
    joblib.dump(scaler, "scaler.joblib")

def load_model():
    if os.path.exists("risk_model.joblib") and os.path.exists("scaler.joblib"):
        model = joblib.load("risk_model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    return None, None

def main():
    st.set_page_config(page_title="Customer Risk Profiler", layout="wide")
    st.title("📊 Customer Risk Profiling Model")

    st.sidebar.header("📁 Choose Data Source")
    mode = st.sidebar.radio("Select mode", ["Use SQL Database", "Upload CSV for Training", "Upload CSV for Prediction Only"])

    model, scaler = load_model()
    features = []

    if mode == "Use SQL Database":
        try:
            df = load_data()
            if df is not None:
                st.write("### Sample Data from SQL", df.head())
                X_train, X_test, y_train, y_test, scaler, features = preprocess(df)
                model = train_model(X_train, y_train)
                save_model(model, scaler)

                if st.button("📊 Show Model Stats"):
                    st.subheader("📈 Model Evaluation")
                    evaluate_model(model, X_test, y_test)

                predict_uploaded(model, scaler, features)
        except Exception as e:
            st.error(f"❌ Failed to load data from database: {e}")

    elif mode == "Upload CSV for Training":
        uploaded_train = st.file_uploader("Upload CSV with labels", type=["csv"], key="train_csv")
        if uploaded_train:
            df = pd.read_csv(uploaded_train)
            st.write("📋 Training Data Preview", df.head())

            df.fillna(0, inplace=True)

            if "total_paid" in df.columns and "total_due" in df.columns:
                df["payment_ratio"] = df["total_paid"] / (df["total_due"] + 1)
            if "total_delay_days" in df.columns and "num_late_payments" in df.columns:
                df["avg_delay_days"] = df["total_delay_days"] / (df["num_late_payments"] + 1)

            suggest_features(df)

            target = "default" if "default" in df.columns else df.columns[-1]
            features = [col for col in df.columns if col not in ["customer_id", "name", target]]
            X = df[features]
            y = df[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = train_model(X_scaled, y)
            save_model(model, scaler)
            st.success(f"✅ Model trained using target: `{target}`")

            if st.button("📊 Show Model Stats"):
                st.subheader("📈 Model Evaluation")
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                evaluate_model(model, X_test, y_test)

    elif mode == "Upload CSV for Prediction Only":
        if model is None or scaler is None:
            st.warning("⚠️ No model found. Please train one first.")
        else:
            predict_uploaded(model, scaler, None)

    collect_feedback()

if __name__ == "__main__":
    main()
