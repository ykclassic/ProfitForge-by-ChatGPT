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
            df.columns = [c.lower().strip() for c in df.columns]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"DB Load Error: {e}")
        return pd.DataFrame()

def get_stats(df):
    if df.empty or 'outcome' not in df.columns:
        return "0%", 0
    # Filter for finalized outcomes
    finished = df[df['outcome'].isin(['TAKE_PROFIT', 'STOP_LOSS'])]
    if finished.empty:
        return "Pending", 0
    wins = len(finished[finished['outcome'] == 'TAKE_PROFIT'])
    rate = (wins / len(finished)) * 100
    return f"{rate:.1f}%", len(finished)

# --- UI Layout ---
st.title("📈 Nexus Predictive Command Center")

if os.path.exists(DB_PATH):
    df = load_data(os.path.getmtime(DB_PATH))
    
    if not df.empty:
        # 1. KPI Header
        win_rate, total_trades = get_stats(df)
        latest_batch = df.sort_values('timestamp').groupby('symbol').tail(1)
        top_conf = latest_batch.loc[latest_batch['confidence'].idxmax()]

        k1, k2, k3 = st.columns(3)
        k1.metric("Current Win Rate", win_rate, help="Calculated from TAKE_PROFIT vs STOP_LOSS hits")
        k2.metric("Top Confidence", f"{top_conf['symbol']}", f"{top_conf['confidence']:.1%}")
        k3.metric("Total Signals", len(df))

        st.divider()

        # 2. Live Forecast Table
        st.subheader("📡 Live Market Forecasts")
        # Format the table for readability
        display_df = df.copy()
        display_df['entry'] = display_df['entry'].map('${:,.4f}'.format)
        display_df['tp'] = display_df['tp'].map('${:,.4f}'.format)
        display_df['sl'] = display_df['sl'].map('${:,.4f}'.format)
        
        # Color coding the outcomes
        def color_outcome(val):
            color = '#2ecc71' if val == 'TAKE_PROFIT' else '#e74c3c' if val == 'STOP_LOSS' else '#f1c40f'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            display_df[['timestamp', 'symbol', 'signal_type', 'confidence', 'entry', 'tp', 'sl', 'outcome']].style.applymap(color_outcome, subset=['outcome']),
            use_container_width=True,
            hide_index=True
        )

        # 3. Sidebar Analytics
        st.sidebar.header("Asset Intelligence")
        selected = st.sidebar.selectbox("Focus Asset", df['symbol'].unique())
        asset_history = df[df['symbol'] == selected].sort_values('timestamp')
        
        st.sidebar.divider()
        st.sidebar.write(f"### {selected} Momentum")
        st.sidebar.line_chart(asset_history.set_index('timestamp')['confidence'])
    else:
        st.info("Database is synced but currently empty. Running the next cycle...")
else:
    st.warning("Awaiting initial database sync from GitHub Actions.")

if st.button("Force Global Refresh"):
    st.cache_data.clear()
    st.rerun()
