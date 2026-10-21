import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Daily Demand Forecasting", page_icon="📦")
st.title("📦 Daily Demand Forecasting")
st.write("Enter the details of a day to predict the **total number of orders**.")

col1, col2 = st.columns(2)
with col1:
    week = st.selectbox("Week of the month", [1, 2, 3, 4, 5], index=2)
    day = st.selectbox("Day of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    non_urgent = st.number_input("Non-urgent orders", min_value=0.0, value=150.0)
    urgent = st.number_input("Urgent orders", min_value=0.0, value=110.0)
    fiscal = st.number_input("Fiscal sector orders", min_value=0.0, value=10.0)
with col2:
    traffic = st.number_input("Traffic controller sector orders", min_value=0.0, value=40000.0)
    bank1 = st.number_input("Banking orders (1)", min_value=0.0, value=50000.0)
    bank2 = st.number_input("Banking orders (2)", min_value=0.0, value=70000.0)
    bank3 = st.number_input("Banking orders (3)", min_value=0.0, value=20000.0)

if st.button("Predict"):
    payload = {
        "week_of_month": week, "day_of_week": day,
        "non_urgent_orders": non_urgent, "urgent_orders": urgent,
        "fiscal_sector_orders": fiscal, "traffic_controller_orders": traffic,
        "banking_orders_1": bank1, "banking_orders_2": bank2, "banking_orders_3": bank3,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
        if r.status_code == 200:
            res = r.json()
            st.success(f"Predicted total orders: **{res['predicted_total_orders']}**")
            low, high = res["approx_range"]
            st.info(f"Typical range: {low} – {high}   |   Model version: {res['model_version']}")
        else:
            st.error(f"API error: {r.json().get('detail', r.text)}")
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API at {API_URL}. ({e})")
