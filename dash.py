import streamlit as st
import sqlite3
import pandas as pd
import os

# --- Lead Developer Config ---
st.set_page_config(layout="wide", page_title="Nexus Predictive", page_icon="🔮")

DB_PATH = "data/trading.db"

@st.cache_data(ttl=30)
def load_data(mtime):
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
    conn.close()
    return df

st.title("🔮 Nexus Predictive Forecasting")

if os.path.exists(DB_PATH):
    signals = load_data(os.path.getmtime(DB_PATH))
    if not signals.empty:
        latest = signals.iloc[0]
        
        # Predictive Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Direction", latest['signal_type'])
        c2.metric("Confidence", f"{latest['confidence']:.2%}")
        # Displaying the predicted magnitude from our Regressor
        if 'pred_move' in latest:
            c3.metric("Forecasted Move", f"±{latest['pred_move']:.2%}")
        c4.metric("Regime", f"State {latest['regime']}")

        st.divider()
        
        st.subheader("Forecasted Trade Levels")
        t1, t2, t3 = st.columns(3)
        t1.info(f"Entry: **${latest['entry']:,.2f}**")
        t2.error(f"Stop Loss: **${latest['sl']:,.2f}**")
        t3.success(f"Take Profit: **${latest['tp']:,.2f}**")
        
        st.subheader("Historical Accuracy (Confidence Trend)")
        st.line_chart(signals.set_index('timestamp')['confidence'])
        
        st.subheader("Full Nexus Logs")
        st.dataframe(signals, use_container_width=True)
    else:
        st.info("Synchronizing with GitHub Actions...")
else:
    st.warning("Database not found. Trigger the GitHub Action to begin.")
