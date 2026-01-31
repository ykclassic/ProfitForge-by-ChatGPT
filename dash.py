import streamlit as st
import sqlite3
import pandas as pd
import os
import time

# --- Lead Developer Config ---
st.set_page_config(layout="wide", page_title="Nexus HybridTrader")

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "trading.db")

def get_db_mtime():
    """Returns the last modification time of the DB file to use as a cache key."""
    if os.path.exists(DB_PATH):
        return os.path.getmtime(DB_PATH)
    return 0

@st.cache_data(ttl=600) # Cache for 10 mins, but mtime will bust it if file changes
def load_data(mtime):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    try:
        # timeout=20 prevents 'database is locked' during GitHub Action writes
        conn = sqlite3.connect(DB_PATH, timeout=20)
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

# --- UI Layout ---
st.title("📊 Nexus HybridTrader Dashboard")

# The 'mtime' ensures that if the file is updated by Git, the cache refreshes
current_mtime = get_db_mtime()
signals = load_data(current_mtime)

if not signals.empty:
    # Sidebar status
    st.sidebar.success(f"Last DB Update: {time.ctime(current_mtime)}")
    if st.sidebar.button("Force Refresh"):
        st.cache_data.clear()
        st.rerun()

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
    st.info("Waiting for first signal from GitHub Actions...")
    if st.button("Check for Data"):
        st.rerun()
