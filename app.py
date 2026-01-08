import streamlit as st
import plotly.express as px

from modules.data_collection import fetch_crypto_data
from modules.preprocessing import preprocess_data
from modules.volatility import calculate_volatility
from modules.arima_model import arima_forecast
from modules.prophet_model import prophet_forecast

st.set_page_config(layout="wide")
st.title("📊 Cryptocurrency Time Series Analysis & Forecasting")

coin = st.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "dogecoin"])
df = fetch_crypto_data(coin)
df = preprocess_data(df)

# ---- PRICE CHART ----
st.subheader("Historical Price Trend")
fig = px.line(df, x=df.index, y="price")
st.plotly_chart(fig, use_container_width=True)

# ---- VOLATILITY ----
vol = calculate_volatility(df)
st.metric("Annualized Volatility", f"{vol:.2%}")

# ---- MOVING AVERAGES ----
st.subheader("Moving Averages")
fig2 = px.line(df, y=["price", "ma_7", "ma_30"])
st.plotly_chart(fig2, use_container_width=True)

# ---- ARIMA ----
st.subheader("ARIMA Forecast")
if st.button("Run ARIMA"):
    forecast = arima_forecast(df["price"])
    st.line_chart(forecast)

# ---- PROPHET ----
st.subheader("Prophet Forecast")
if st.button("Run Prophet"):
    pf = prophet_forecast(df)
    st.line_chart(pf.set_index("ds")["yhat"])
