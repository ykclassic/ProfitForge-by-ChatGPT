import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime

# --- Config ---
st.set_page_config(layout="wide", page_title="Nexus Command Center", page_icon="📈")
DB_PATH = "data/trading.db"

def repair_database_schema():
    """Lead Dev Fix: Ensures the DB has all required columns (symbol, status, etc.)."""
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(signals)")
    columns = [info[1].lower() for info in cursor.fetchall()]
    
    # Required columns for the new features
    required = {
        "symbol": "TEXT", 
        "status": "TEXT DEFAULT 'PENDING'", 
        "pred_move": "REAL",
        "confidence": "REAL"
    }
    
    for col, col_type in required.items():
        if col not in columns:
            cursor.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()

@st.cache_data(ttl=15)
def load_data(mtime):
    repair_database_schema()
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        # Pull the last 200 signals to ensure we have a rich history for charts
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 200", conn)
        conn.close()
        
        if not df.empty:
            df.columns = [c.lower().strip() for c in df.columns]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Ensure numbers are treated as floats for calculations
            for col in ['confidence', 'entry', 'sl', 'tp', 'pred_move']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

def calculate_win_rate(df):
    if 'status' not in df.columns or df.empty:
        return "N/A"
    outcomes = df[df['status'].str.contains('HIT', na=False, case=False)]
    if outcomes.empty:
        return "0%"
    wins = len(outcomes[outcomes['status'].str.contains('TAKE PROFIT', case=False)])
    return f"{(wins / len(outcomes)) * 100:.1f}%"

# --- UI Execution ---
st.title("📈 Nexus Alpha Command Center")

if os.path.exists(DB_PATH):
    df = load_data(os.path.getmtime(DB_PATH))
    
    if not df.empty and 'symbol' in df.columns:
        # Robust selection: Get the very last signal entry for each unique symbol
        latest_batch = df.sort_values('timestamp').groupby('symbol').tail(1)
        
        if not latest_batch.empty:
            # Find the signal with the highest confidence
            top_pick = latest_batch.loc[latest_batch['confidence'].idxmax()]
            
            # --- KPI Section ---
            k1, k2, k3 = st.columns(3)
            k1.metric("Current Win Rate", calculate_win_rate(df))
            k2.metric("Top Asset", top_pick['symbol'], f"{top_pick['confidence']:.1%} Conf")
            k3.metric("Last Run (UTC)", df['timestamp'].max().strftime('%H:%M'))

            st.divider()

            # --- Focus Section ---
            st.subheader(f"🎯 Immediate Opportunity: {top_pick['symbol']}")
            f1, f2, f3, f4 = st.columns(4)
            f1.info(f"Side: **{top_pick['signal_type']}**")
            f2.write(f"Entry: **${top_pick['entry']:,.4f}**")
            f3.error(f"SL: **${top_pick['sl']:,.4f}**")
            f4.success(f"TP: **${top_pick['tp']:,.4f}**")

            # --- Signal History Table ---
            st.subheader("Signal Log & Historical Outcomes")
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # --- Interactive Analytics ---
            st.sidebar.header("Deep Dive")
            selected = st.sidebar.selectbox("Select Asset", df['symbol'].unique())
            asset_df = df[df['symbol'] == selected].sort_values('timestamp')
            
            st.sidebar.metric(f"Current {selected} Status", asset_df.iloc[-1]['status'])
            
            st.write(f"### {selected} Momentum Analysis")
            c1, c2 = st.columns(2)
            c1.line_chart(asset_df.set_index('timestamp')['confidence'])
            if 'pred_move' in asset_df.columns:
                c2.area_chart(asset_df.set_index('timestamp')['pred_move'])
        else:
            st.info("Database initialized. Waiting for the next hourly trainer run to populate symbols.")
    else:
        st.warning("Database found but no signal data detected. Check your GitHub Actions logs for errors.")
else:
    st.error(f"Database missing at {DB_PATH}. Run your trainer script at least once.")

if st.button("🔄 Sync Live Data"):
    st.cache_data.clear()
    st.rerun()
