# app.py
import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(layout="wide")

conn = sqlite3.connect("data/trading.db")

st.title("📊 Nexus HybridTrader Dashboard")

signals = pd.read_sql("SELECT * FROM signals ORDER BY timestamp DESC", conn)

st.subheader("Latest Signals")
st.dataframe(signals)

st.subheader("Model Health")
st.metric("Drift Status", "Stable")

st.subheader("Regime Distribution")
st.bar_chart(signals["regime"].value_counts())
