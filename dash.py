import streamlit as st
import sqlite3
import pandas as pd
import os

# --- Config ---
st.set_page_config(layout="wide", page_title="Nexus Command Center", page_icon="📈")
DB_PATH = "data/trading.db"

@st.cache_data(ttl=30)
def load_data(mtime):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not df.empty:
            # SAFETY FIX: Force all column names to lowercase and remove extra spaces
            df.columns = [c.lower().strip() for c in df.columns]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure 'status' and 'symbol' exist to prevent KeyErrors
            if 'status' not in df.columns:
                df['status'] = "PENDING"
            if 'symbol' not in df.columns:
                st.error("Critical Error: 'symbol' column missing from database.")
                return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Database Read Error: {e}")
        return pd.DataFrame()

def calculate_win_rate(df):
    if 'status' not in df.columns or df.empty:
        return "N/A"
    # Filter for finalized trades
    outcomes = df[df['status'].str.contains('HIT', na=False, case=False)]
    if outcomes.empty:
        return "0%"
    wins = len(outcomes[outcomes['status'].str.contains('TAKE PROFIT', case=False)])
    rate = (wins / len(outcomes)) * 100
    return f"{rate:.1f}%"

# --- Dashboard UI ---
st.title("📈 Nexus Multi-Asset Command Center")

if os.path.exists(DB_PATH):
    df = load_data(os.path.getmtime(DB_PATH))
    
    if not df.empty and 'symbol' in df.columns:
        # 1. Performance KPI Bar
        # Get the most recent signal for each unique symbol
        latest_batch = df.sort_values('timestamp').groupby('symbol').tail(1)
        
        # Find Top Pick based on confidence
        top_pick = latest_batch.loc[latest_batch['confidence'].idxmax()]
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Global Win Rate", calculate_win_rate(df))
        kpi2.metric("Top Alpha Pick", top_pick['symbol'], delta=f"{top_pick['confidence']:.1%}")
        kpi3.metric("Monitored Assets", len(latest_batch))

        st.divider()

        # 2. Alpha Alert Spotlight
        st.subheader(f"🔥 Top Pick Spotlight: {top_pick['symbol']}")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("Direction", top_pick['signal_type'])
        t_col2.info(f"Entry: **${top_pick['entry']:,.4f}**")
        t_col3.error(f"Stop Loss: **${top_pick['sl']:,.4f}**")
        t_col4.success(f"Take Profit: **${top_pick['tp']:,.4f}**")

        st.divider()

        # 3. Main Forecast Table
        st.subheader("Live Market Forecasts & Outcome History")
        cols_to_show = ['timestamp', 'symbol', 'signal_type', 'confidence', 'entry', 'sl', 'tp', 'status']
        # Filter only existing columns to be safe
        available_cols = [c for c in cols_to_show if c in df.columns]
        
        st.dataframe(
            df[available_cols],
            use_container_width=True,
            hide_index=True
        )

        # 4. Sidebar Deep Dive
        st.sidebar.header("Navigation")
        selected_asset = st.sidebar.selectbox("Select Asset for Deep Dive", df['symbol'].unique())
        asset_df = df[df['symbol'] == selected_asset]
        
        st.sidebar.divider()
        st.sidebar.write(f"**Last {selected_asset} Signal:**")
        st.sidebar.code(f"{asset_df.iloc[0]['signal_type']} @ {asset_df.iloc[0]['entry']}")

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"### {selected_asset} Confidence History")
            st.line_chart(asset_df.set_index('timestamp')['confidence'])
        with c2:
            st.write(f"### {selected_asset} Predicted Volatility")
            if 'pred_move' in asset_df.columns:
                st.area_chart(asset_df.set_index('timestamp')['pred_move'])

    else:
        st.info("System Initializing: Waiting for first multi-asset batch to populate database.")
else:
    st.warning("Database not found. Please ensure the trainer_daemon has run successfully.")

if st.button("Force Refresh"):
    st.cache_data.clear()
    st.rerun()
