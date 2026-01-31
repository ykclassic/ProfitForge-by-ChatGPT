import streamlit as st
import sqlite3
import pandas as pd
import os

# --- Config ---
st.set_page_config(layout="wide", page_title="Nexus Command Center", page_icon="📈")
DB_PATH = "data/trading.db"

def repair_database_schema():
    """Lead Dev Fix: Ensures the DB has all required columns before the app starts."""
    if not os.path.exists(DB_PATH):
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(signals)")
    columns = [info[1].lower() for info in cursor.fetchall()]
    
    required = {
        "symbol": "TEXT",
        "status": "TEXT DEFAULT 'PENDING'",
        "pred_move": "REAL",
        "regime": "INTEGER"
    }
    
    for col, col_type in required.items():
        if col not in columns:
            try:
                cursor.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_type}")
                st.toast(f"✅ Database Migrated: Added {col}")
            except Exception as e:
                st.error(f"Migration Error: {e}")
                
    conn.commit()
    conn.close()

@st.cache_data(ttl=30)
def load_data(mtime):
    repair_database_schema()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)
        conn.close()
        
        if not df.empty:
            df.columns = [c.lower().strip() for c in df.columns]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Ensure confidence is numeric for idxmax
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

def calculate_win_rate(df):
    if 'status' not in df.columns or df.empty:
        return "0%"
    outcomes = df[df['status'].str.contains('HIT', na=False, case=False)]
    if outcomes.empty:
        return "0%"
    wins = len(outcomes[outcomes['status'].str.contains('TAKE PROFIT', case=False)])
    return f"{(wins / len(outcomes)) * 100:.1f}%"

# --- Dashboard Header ---
st.title("📈 Nexus Multi-Asset Command Center")

if os.path.exists(DB_PATH):
    df = load_data(os.path.getmtime(DB_PATH))
    
    # CRITICAL FIX: Only attempt processing if df is NOT empty and has 'symbol'
    if not df.empty and 'symbol' in df.columns:
        latest_batch = df.sort_values('timestamp').groupby('symbol').tail(1)
        
        # GUARD: Ensure latest_batch is not empty before calling idxmax
        if not latest_batch.empty:
            top_pick_idx = latest_batch['confidence'].idxmax()
            top_pick = latest_batch.loc[top_pick_idx]
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Global Win Rate", calculate_win_rate(df))
            k2.metric("Top Alpha Pick", top_pick['symbol'], f"{top_pick['confidence']:.1%}")
            k3.metric("Assets Tracked", len(latest_batch))

            st.divider()

            # Spotlight
            st.subheader(f"🔥 Spotlight: {top_pick['symbol']} ({top_pick['signal_type']})")
            s1, s2, s3 = st.columns(3)
            s1.info(f"Entry: **${top_pick['entry']:,.4f}**")
            s2.error(f"Stop Loss: **${top_pick['sl']:,.4f}**")
            s3.success(f"Take Profit: **${top_pick['tp']:,.4f}**")

            st.subheader("Live Forecasts & Outcomes")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Sidebar Analytics
            st.sidebar.header("Analytics")
            asset = st.sidebar.selectbox("Deep Dive", df['symbol'].unique())
            asset_df = df[df['symbol'] == asset].sort_values('timestamp')
            
            st.subheader(f"📊 {asset} Trends")
            c1, c2 = st.columns(2)
            c1.line_chart(asset_df.set_index('timestamp')['confidence'])
            if 'pred_move' in asset_df.columns:
                c2.area_chart(asset_df.set_index('timestamp')['pred_move'])
        else:
            st.info("Gathering data... No recent signals found in batch.")
    else:
        st.warning("Database is empty. Please run the trainer script to generate signals.")
else:
    st.error(f"Database file not found at {DB_PATH}. Check your data folder.")

if st.button("Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()
