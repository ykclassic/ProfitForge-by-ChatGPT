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

@st.cache_data(ttl=60)
def load_data(mtime):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=20)
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
        conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

# --- UI Layout ---
st.title("📊 Nexus HybridTrader Dashboard")

current_mtime = get_db_mtime()
signals = load_data(current_mtime)

if not signals.empty:
    # Header Metrics
    latest = signals.iloc[0]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Entry", f"${latest['entry']:,.2f}")
    
    # Color logic for Signal Type
    signal_color = "normal" if latest['signal_type'] == "LONG" else "inverse"
    m2.metric("Signal", latest['signal_type'], delta=None)
    
    m3.metric("Stop Loss", f"${latest['sl']:,.2f}")
    m4.metric("Take Profit", f"${latest['tp']:,.2f}")

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Signal History & Confidence Trend")
        # Visualizing confidence over time
        chart_data = signals.set_index('timestamp')['confidence'].sort_index()
        st.line_chart(chart_data)
        
        st.subheader("Detailed Logs")
        # Prettifying the table for display
        display_df = signals.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%m-%d %H:%M')
        st.dataframe(display_df, use_container_width=True)

    with col2:
        st.subheader("Model Insights")
        st.write(f"**Current Confidence:** {latest['confidence']:.2%}")
        st.progress(latest['confidence'])
        
        st.subheader("Regime Distribution")
        regime_counts = signals["regime"].value_counts().sort_index()
        st.bar_chart(regime_counts)
        
        if st.button("Clear Cache & Refresh"):
            st.cache_data.clear()
            st.rerun()
else:
    st.info("Waiting for the next GitHub Action cycle to populate enhanced data...")
