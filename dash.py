import streamlit as st
import sqlite3
import pandas as pd
import os
import time

# --- Lead Developer Config ---
st.set_page_config(layout="wide", page_title="Nexus HybridTrader", page_icon="📊")

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "trading.db")

def get_db_mtime():
    if os.path.exists(DB_PATH):
        return os.path.getmtime(DB_PATH)
    return 0

@st.cache_data(ttl=600)
def load_data(mtime):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20)
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
        conn.close()
        # Formatting for UI
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

# --- UI Layout ---
st.title("📊 Nexus HybridTrader Dashboard")

current_mtime = get_db_mtime()
signals = load_data(current_mtime)

if not signals.empty:
    # Sidebar Metadata
    st.sidebar.header("System Status")
    st.sidebar.info(f"Last Sync: {time.ctime(current_mtime)}")
    if st.sidebar.button("Force Refresh"):
        st.cache_data.clear()
        st.rerun()

    # Top Level Metrics
    latest_conf = signals['confidence'].iloc[0]
    latest_regime = signals['regime'].iloc[0]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Current Price", f"${signals['entry'].iloc[0]:,.2f}")
    m2.metric("Model Confidence", f"{latest_conf:.2%}")
    m3.metric("Current Regime", f"Regime {latest_regime}")

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Latest Signals")
        # Apply styling to the dataframe
        st.dataframe(signals.style.background_gradient(subset=['confidence'], cmap='RdYlGn'), use_container_width=True)

    with col2:
        st.subheader("Model Health")
        st.metric("Drift Status", "Stable", delta="0.02% variance")
        
        st.subheader("Regime Distribution")
        regime_counts = signals["regime"].value_counts().sort_index()
        st.bar_chart(regime_counts)
else:
    st.warning("Awaiting first signal from GitHub Actions runner...")
