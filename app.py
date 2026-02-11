import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stock Direction Predictor", layout="centered")

st.title("📈 Real-Time Stock Direction Predictor")
st.write("Predict whether stock price will go UP or DOWN tomorrow")

# ---------------------------------------------------
# 50+ Popular Stock Symbols (US + India Mix)
# ---------------------------------------------------

stock_list = [
    # US Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
    "INTC", "AMD", "ORCL", "IBM", "ADBE", "CSCO", "QCOM",

    # US Companies
    "WMT", "KO", "PEP", "DIS", "BA", "JPM", "GS", "V", "MA", "PYPL",
    "XOM", "CVX", "NKE", "MCD", "T",

    # Indian Stocks (.NS required for NSE)
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "ITC.NS", "WIPRO.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "ONGC.NS",
    "ADANIENT.NS", "POWERGRID.NS"
]

# Dropdown selection
stock_symbol = st.selectbox("Select Stock Symbol", stock_list)

if st.button("Predict"):

    with st.spinner("Fetching stock data..."):

        data = yf.download(stock_symbol, period="2y")

    if data.empty:
        st.error("Unable to fetch data.")
    else:
        # Feature Engineering
        data["Return"] = data["Close"].pct_change()
        data["MA10"] = data["Close"].rolling(10).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["Volatility"] = data["Return"].rolling(10).std()

        # Target variable
        data["Target"] = np.where(data["Return"].shift(-1) > 0, 1, 0)

        data = data.dropna()

        features = ["MA10", "MA50", "Volatility"]
        X = data[features]
        y = data["Target"]

        # Train/Test split (no shuffle for time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Accuracy
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Tomorrow prediction
        latest_features = X.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(latest_features)[0]
        prob = model.predict_proba(latest_features)[0][prediction]

        # Results
        st.subheader("📊 Model Accuracy")
        st.info(f"{round(acc * 100, 2)}%")

        st.subheader("🔮 Tomorrow's Prediction")

        if prediction == 1:
            st.success(f"📈 UP (Confidence: {round(prob*100,2)}%)")
        else:
            st.error(f"📉 DOWN (Confidence: {round(prob*100,2)}%)")

        # Chart
        st.subheader("📉 Closing Price Chart")

        fig, ax = plt.subplots()
        ax.plot(data["Close"])
        ax.set_title(f"{stock_symbol} Closing Price (2 Years)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)
