import streamlit as st
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="Nexus Command", layout="wide")

DB_PATH = "data/trading.db"

def get_data():
    if not os.path.exists(DB_PATH): return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 50", conn)
        conn.close()
        return df
    except: return None

st.title("🛰️ Nexus Live Intelligence")

df = get_data()

if df is not None and not df.empty:
    st.metric("Latest Signal", df.iloc[0]['symbol'], df.iloc[0]['signal_type'])
    st.dataframe(df, use_container_width=True)
else:
    # Prevents blank screen
    st.warning("📡 **Connecting to Nexus Grid...**")
    st.info("The database is currently synchronizing with GitHub Actions. Refresh in 60 seconds.")
    if st.button("Refresh Now"):
        st.rerun()
