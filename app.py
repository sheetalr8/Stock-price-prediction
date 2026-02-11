import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

st.set_page_config(page_title="AI Investment Dashboard", layout="wide")

st.title("📈 AI Investment Advisor Dashboard")

# ------------------------------
# Sidebar Controls
# ------------------------------

st.sidebar.header("Controls")

stock_list = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
    "INTC", "AMD", "ORCL", "IBM", "ADBE", "CSCO", "QCOM",
    "WMT", "KO", "PEP", "DIS", "BA", "JPM", "GS", "V", "MA",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "LT.NS", "ITC.NS"
]

stock_symbol = st.sidebar.selectbox("Select Stock", stock_list)

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

# ------------------------------
# Download Data
# ------------------------------

data = yf.download(stock_symbol, period="2y")

# ------------------------------
# Feature Engineering
# ------------------------------

data["Return"] = data["Close"].pct_change()
data["MA10"] = data["Close"].rolling(10).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["Volatility"] = data["Return"].rolling(10).std()

# RSI Calculation
delta = data["Close"].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data["RSI"] = 100 - (100 / (1 + rs))

# Target
data["Target"] = np.where(data["Return"].shift(-1) > 0, 1, 0)

data = data.dropna()

features = ["MA10", "MA50", "Volatility", "RSI"]
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ------------------------------
# Model Selection
# ------------------------------

if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Random Forest":
    model = RandomForestClassifier()
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Tomorrow Prediction
latest_features = X.iloc[-1].values.reshape(1, -1)
prediction = model.predict(latest_features)[0]
probability = model.predict_proba(latest_features)[0][prediction]

# ------------------------------
# Layout Columns
# ------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Accuracy")
    st.info(f"{round(accuracy*100,2)} %")

with col2:
    st.subheader("🔮 Tomorrow Prediction")

    if prediction == 1:
        st.success(f"📈 UP ({round(probability*100,2)}% confidence)")
    else:
        st.error(f"📉 DOWN ({round(probability*100,2)}% confidence)")

# ------------------------------
# Buy / Sell Logic
# ------------------------------

st.subheader("💡 AI Recommendation")

if probability > 0.7 and prediction == 1:
    st.success("🟢 Strong BUY Signal")
elif probability > 0.6 and prediction == 1:
    st.info("🟡 BUY")
elif probability > 0.6 and prediction == 0:
    st.warning("🟠 SELL")
else:
    st.error("🔴 Strong SELL Signal")

# ------------------------------
# Interactive Plotly Chart
# ------------------------------

st.subheader("📉 Interactive Stock Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
))

fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=600
)

st.plotly_chart(fig, use_container_width=True)
