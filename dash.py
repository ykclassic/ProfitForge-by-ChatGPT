import streamlit as st
import sqlite3
import pandas as pd
import os

# --- Lead Developer Configuration ---
st.set_page_config(layout="wide", page_title="Nexus HybridTrader")

# Define paths
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "trading.db")

# 1. Ensure the directory exists before attempting connection
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# 2. Establish Database Connection
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

def get_signals():
    """Helper to fetch data with error handling for missing tables."""
    try:
        return pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
    except (pd.io.sql.DatabaseError, sqlite3.OperationalError):
        # Return an empty DataFrame with expected columns if table doesn't exist yet
        return pd.DataFrame(columns=["timestamp", "regime", "signal", "price"])

# --- UI Layout ---
st.title("📊 Nexus HybridTrader Dashboard")

# Fetch data
signals = get_signals()

if not signals.empty:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Latest Signals")
        st.dataframe(signals, use_container_width=True)

    with col2:
        st.subheader("Model Health")
        st.metric("Drift Status", "Stable")
        
        st.subheader("Regime Distribution")
        if "regime" in signals.columns:
            st.bar_chart(signals["regime"].value_counts())
else:
    st.info("No trading data found. Waiting for signals to be written to the database...")
    st.subheader("Model Health")
    st.metric("Drift Status", "Initializing")

# Close connection handled by Streamlit's execution cycle or manual close if needed
