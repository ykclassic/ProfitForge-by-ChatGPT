import streamlit as st
import sqlite3
import pandas as pd
import os

# --- Lead Developer Config ---
st.set_page_config(layout="wide", page_title="Nexus Command Center", page_icon="📈")

DB_PATH = "data/trading.db"

@st.cache_data(ttl=30)
def load_data(mtime):
    """Loads all historical signals from the SQLite database."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
        conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Database Read Error: {e}")
        return pd.DataFrame()

def calculate_win_rate(df):
    """Calculates global win rate based on the 'Status' column."""
    if 'status' not in df.columns or df.empty:
        return "N/A"
    outcomes = df[df['status'].str.contains('HIT', na=False)]
    if outcomes.empty:
        return "0%"
    wins = len(outcomes[outcomes['status'].str.contains('TAKE PROFIT')])
    rate = (wins / len(outcomes)) * 100
    return f"{rate:.1f}%"

# --- Dashboard Header ---
st.title("📈 Nexus Multi-Asset Command Center")

if os.path.exists(DB_PATH):
    # mtime is passed to the cache function to trigger refresh on file update
    df = load_data(os.path.getmtime(DB_PATH))
    
    if not df.empty:
        # 1. High-Level KPIs
        latest_batch = df.groupby('symbol').first().reset_index()
        top_pick = latest_batch.loc[latest_batch['confidence'].idxmax()]
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Global Win Rate", calculate_win_rate(df))
        with kpi2:
            st.metric("Top Alpha Pick", top_pick['symbol'], delta=f"{top_pick['confidence']:.1%}")
        with kpi3:
            st.metric("Active Assets", len(latest_batch))

        st.divider()

        # 2. Top Pick Spotlight
        st.subheader(f"🔥 Current Alpha Signal: {top_pick['symbol']}")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Action", top_pick['signal_type'])
        t_col2.info(f"Entry: **${top_pick['entry']:,.4f}**")
        t_col3.error(f"Stop Loss: **${top_pick['sl']:,.4f}**")
        t_col4.success(f"Take Profit: **${top_pick['tp']:,.4f}**")

        st.divider()

        # 3. All Active Signals & Historical Logs
        st.subheader("Predictive Forecasts & Outcomes")
        
        # Ensure 'status' column exists for display even if empty in DB
        if 'status' not in df.columns:
            df['status'] = "PENDING"

        # Interactive Table with Search/Filter
        st.dataframe(
            df[['timestamp', 'symbol', 'signal_type', 'confidence', 'entry', 'sl', 'tp', 'status']],
            use_container_width=True,
            hide_index=True
        )

        # 4. Asset Deep Dive Section
        st.sidebar.header("Asset Selection")
        selected = st.sidebar.selectbox("Filter Deep Dive", df['symbol'].unique())
        
        asset_df = df[df['symbol'] == selected]
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader(f"{selected} Confidence Trend")
            st.line_chart(asset_df.set_index('timestamp')['confidence'])
        with col_right:
            st.subheader(f"{selected} Forecasted Move")
            if 'pred_move' in asset_df.columns:
                st.area_chart(asset_df.set_index('timestamp')['pred_move'])

        if st.button("Manual Refresh"):
            st.cache_data.clear()
            st.rerun()
    else:
        st.info("System Initializing: Waiting for first multi-asset batch...")
else:
    st.warning("Database not found. Please trigger the GitHub Action to generate the trading database.")
