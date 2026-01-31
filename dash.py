import streamlit as st
import sqlite3
import pandas as pd
import os
import time

# --- Lead Developer Config ---
st.set_page_config(layout="wide", page_title="Nexus HybridTrader", page_icon="📊")

DB_PATH = "data/trading.db"

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

st.title("📊 Nexus HybridTrader Dashboard")

signals = load_data(os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0)

if not signals.empty:
    latest = signals.iloc[0]
    
    # 1. Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Entry Price", f"${latest['entry']:,.2f}")
    
    sig_delta = "Bullish" if latest['signal_type'] == "LONG" else "Bearish"
    m2.metric("Signal Type", latest['signal_type'], delta=sig_delta)
    
    m3.metric("Stop Loss (ATR Adj)", f"${latest['sl']:,.2f}")
    m4.metric("Take Profit (ATR Adj)", f"${latest['tp']:,.2f}")

    st.divider()

    # 2. Main Analytics
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Confidence & Volatility Trend")
        # Line chart showing Confidence vs ATR
        st.line_chart(signals.set_index('timestamp')[['confidence', 'atr']])
        
        st.subheader("Signal Log")
        st.dataframe(signals.style.highlight_max(axis=0, subset=['confidence']), use_container_width=True)

    with col2:
        st.subheader("Market State (Stable HMM)")
        regime_labels = {0: "Low Vol / Calm", 1: "Medium Vol", 2: "High Vol / Stress"}
        current_label = regime_labels.get(latest['regime'], "Unknown")
        st.info(f"Current State: **{current_label}**")
        
        st.subheader("Regime Distribution")
        st.bar_chart(signals["regime"].value_counts())

        if st.button("Refresh System"):
            st.cache_data.clear()
            st.rerun()
else:
    st.warning("Nexus System Initializing... check back after the next GitHub Action cycle.")
