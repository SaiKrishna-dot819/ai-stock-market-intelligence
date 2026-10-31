
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Stock Market Intelligence",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Market Intelligence")
st.caption(
    "Advanced stock analysis, machine learning, "
    "technical indicators and price forecasting"
)

st.warning(
    "Educational project only. Predictions are uncertain "
    "and must not be treated as guaranteed investment advice."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Market Settings")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="AAPL"
).strip().upper()

market_interval = st.sidebar.selectbox(
    "Data interval",
    ["Daily", "Hourly"]
)

interval = "1d" if market_interval == "Daily" else "1h"

period = st.sidebar.selectbox(
    "Historical data",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

st.sidebar.markdown("""
**Example tickers**

Indian:
- `RELIANCE.NS`
- `TCS.NS`
- `INFY.NS`
- `^NSEI`

International:
- `AAPL`
- `MSFT`
- `NVDA`
- `GOOGL`
""")

# --------------------------------------------------
# DOWNLOAD MARKET DATA
# --------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(symbol, period, interval):
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).title() for c in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    return df


with st.spinner("Downloading stock market data..."):
    stock = get_stock_data(ticker, period, interval)

if stock.empty:
    st.error(
        "No data found. Check the ticker, try another period, "
        "or check whether the data provider is available."
    )
    st.stop()

if len(stock) < 60:
    st.warning(
        "There are fewer than 60 observations. "
        "Forecasting results may be unreliable."
    )

latest = stock.iloc[-1]
previous = stock.iloc[-2] if len(stock) > 1 else latest

price_change = latest["Close"] - previous["Close"]

change_percent = (
    price_change / previous["Close"] * 100
    if previous["Close"] != 0 else 0
)

# --------------------------------------------------
# MARKET SUMMARY
# --------------------------------------------------

st.subheader(f"Market overview: {ticker}")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Latest close",
    f"{latest['Close']:,.2f}",
    f"{price_change:+.2f} ({change_percent:+.2f}%)"
)

c2.metric("Latest high", f"{latest['High']:,.2f}")
c3.metric("Latest low", f"{latest['Low']:,.2f}")

c4.metric(
    "Volume",
    f"{latest['Volume']:,.0f}"
)

c5.metric("Data points", len(stock))

# --------------------------------------------------
# TECHNICAL INDICATORS
# --------------------------------------------------

def calculate_indicators(df):
    df = df.copy()

    close = df["Close"]
    delta = close.diff()

    # Moving averages
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()

    # Exponential moving averages
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    df["MACD_Signal"] = (
        df["MACD"].ewm(span=9, adjust=False).mean()
    )

    # RSI
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()

    relative_strength = gains / losses.replace(0, np.nan)

    df["RSI"] = 100 - (
        100 / (1 + relative_strength)
    )

    # Bollinger bands
    rolling_std = close.rolling(20).std()

    df["BB_Upper"] = df["SMA_20"] + 2 * rolling_std
    df["BB_Lower"] = df["SMA_20"] - 2 * rolling_std

    # Returns and volatility
    df["Return"] = close.pct_change()
    df["Volatility"] = df["Return"].rolling(20).std()

    df["Range_Percent"] = (
        (df["High"] - df["Low"]) /
        close.replace(0, np.nan)
    )

    df["Volume_Change"] = (
        df["Volume"].pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    return df.replace([np.inf, -np.inf], np.nan)


featured = calculate_indicators(stock)

# --------------------------------------------------
# INTERACTIVE PRICE CHART
# --------------------------------------------------

st.subheader("Historical price analysis")

chart = go.Figure()

chart.add_trace(
    go.Candlestick(
        x=featured.index,
        open=featured["Open"],
        high=featured["High"],
        low=featured["Low"],
        close=featured["Close"],
        name="Stock price"
    )
)

chart.add_trace(
    go.Scatter(
        x=featured.index,
        y=featured["SMA_20"],
        name="SMA 20"
    )
)

chart.add_trace(
    go.Scatter(
        x=featured.index,
        y=featured["SMA_50"],
        name="SMA 50"
    )
)

chart.update_layout(
    height=550,
    xaxis_rangeslider_visible=False,
    yaxis_title="Price",
    legend_orientation="h"
)

st.plotly_chart(chart, use_container_width=True)

# High and low chart
hl_chart = go.Figure()

hl_chart.add_trace(
    go.Scatter(
        x=featured.index,
        y=featured["High"],
        name="High",
        mode="lines"
    )
)

hl_chart.add_trace(
    go.Scatter(
        x=featured.index,
        y=featured["Low"],
        name="Low",
        mode="lines"
    )
)

hl_chart.add_trace(
    go.Scatter(
        x=featured.index,
        y=featured["Close"],
        name="Close",
        mode="lines"
    )
)

hl_chart.update_layout(
    title="High, Low and Closing Prices",
    height=350,
    yaxis_title="Price"
)

st.plotly_chart(hl_chart, use_container_width=True)

# Volume chart
volume_chart = go.Figure(
    go.Bar(
        x=featured.index,
        y=featured["Volume"],
        name="Trading volume"
    )
)

volume_chart.update_layout(
    title="Trading Volume",
    height=300
)

st.plotly_chart(volume_chart, use_container_width=True)

# --------------------------------------------------
# MACHINE LEARNING DATASET
# --------------------------------------------------

st.header("🤖 AI Price Prediction")

st.write(
    "The system trains several machine-learning models "
    "and compares their performance on later historical observations."
)

FEATURE_COLUMNS = [
    "Return",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "EMA_12",
    "EMA_26",
    "MACD",
    "MACD_Signal",
    "RSI",
    "BB_Upper",
    "BB_Lower",
    "Volatility",
    "Range_Percent",
    "Volume_Change"
]

# Features at time t predict the next observed bar's close.
dataset = featured[FEATURE_COLUMNS].copy()

dataset["Target"] = stock["Close"].shift(-1)

dataset = dataset.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

if len(dataset) < 60:
    st.error(
        "Not enough usable historical observations for ML. "
        "Try a longer daily period."
    )
    st.stop()

X = dataset[FEATURE_COLUMNS]
y = dataset["Target"]

# Chronological split: no random shuffling.
split = int(len(dataset) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def create_model(name):

    if name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

    elif name == "Extra Trees":
        model = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

    else:
        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42
        )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])


# --------------------------------------------------
# TRAIN AND EVALUATE MODELS
# --------------------------------------------------

model_names = [
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting"
]

results = []
trained_models = {}

with st.spinner("Training machine-learning models..."):

    for name in model_names:

        model = create_model(name)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        # Compare predicted direction with actual direction.
        previous_closes = stock.loc[
            X_test.index, "Close"
        ].values

        actual_direction = np.sign(
            y_test.values - previous_closes
        )

        predicted_direction = np.sign(
            predictions - previous_closes
        )

        directional_accuracy = np.mean(
            actual_direction == predicted_direction
        ) * 100

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "Directional Accuracy (%)": directional_accuracy
        })

        trained_models[name] = model


results_df = pd.DataFrame(results)

results_df = results_df.sort_values("MAE")

st.subheader("Model comparison")

st.dataframe(
    results_df.round(4),
    use_container_width=True,
    hide_index=True
)

best_name = results_df.iloc[0]["Model"]

st.success(
    f"Lowest holdout MAE in this comparison: {best_name}"
)

st.caption(
    "The lowest MAE does not guarantee the best future model. "
    "These metrics are historical, not live performance guarantees."
)

# --------------------------------------------------
# NEXT-BAR PREDICTION
# --------------------------------------------------

best_model = create_model(best_name)

best_model.fit(X, y)

latest_features = featured[FEATURE_COLUMNS].iloc[[-1]]

next_prediction = float(
    best_model.predict(latest_features)[0]
)

current_price = float(stock["Close"].iloc[-1])

predicted_change = next_prediction - current_price

predicted_change_pct = (
    predicted_change / current_price * 100
    if current_price else 0
)

st.subheader("Next observed bar estimate")

p1, p2, p3 = st.columns(3)

p1.metric(
    "Latest close",
    f"{current_price:,.2f}"
)

p2.metric(
    "Predicted next close",
    f"{next_prediction:,.2f}",
    f"{predicted_change:+.2f}"
)

p3.metric(
    "Estimated percentage change",
    f"{predicted_change_pct:+.2f}%"
)

# --------------------------------------------------
# 24-STEP SCENARIO
# --------------------------------------------------

st.subheader("24-step forecast scenario")

st.info(
    "This is an experimental recursive scenario, not a reliable "
    "24-calendar-hour forecast. Daily mode means 24 daily bars; "
    "hourly mode means 24 hourly bars, subject to market closures. "
    "Future volume and price ranges are unknown."
)

simulation = stock.copy()
future_prices = []

steps = 24

for step in range(1, steps + 1):

    simulation_features = calculate_indicators(simulation)

    row = simulation_features[
        FEATURE_COLUMNS
    ].iloc[[-1]]

    prediction = float(
        best_model.predict(row)[0]
    )

    if not np.isfinite(prediction) or prediction <= 0:
        prediction = float(
            simulation["Close"].iloc[-1]
        )

    last_close = float(
        simulation["Close"].iloc[-1]
    )

    last_high = float(
        simulation["High"].iloc[-1]
    )

    last_low = float(
        simulation["Low"].iloc[-1]
    )

    last_volume = float(
        simulation["Volume"].tail(5).median()
    )

    future_prices.append({
        "Step": step,
        "Forecast Price": prediction
    })

    # This simulation holds volume approximately constant and
    # estimates future OHLC. It is not a market simulator.
    if interval == "1h":
        next_time = (
            simulation.index[-1] +
            pd.Timedelta(hours=1)
        )
    else:
        next_time = (
            simulation.index[-1] +
            pd.Timedelta(days=1)
        )

    simulation.loc[next_time] = {
        "Open": last_close,
        "High": max(last_close, prediction, last_high),
        "Low": min(last_close, prediction, last_low),
        "Close": prediction,
        "Volume": last_volume
    }


forecast_df = pd.DataFrame(future_prices)

forecast_chart = go.Figure()

forecast_chart.add_trace(
    go.Scatter(
        x=forecast_df["Step"],
        y=forecast_df["Forecast Price"],
        mode="lines+markers",
        name="Forecast scenario"
    )
)

forecast_chart.update_layout(
    height=400,
    xaxis_title="Future bar number",
    yaxis_title="Estimated price"
)

st.plotly_chart(
    forecast_chart,
    use_container_width=True
)

st.dataframe(
    forecast_df.round(3),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download forecast CSV",
    forecast_df.to_csv(index=False),
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv"
)

# --------------------------------------------------
# TECHNICAL ANALYSIS
# --------------------------------------------------

st.header("📊 Technical Analysis")

last = featured.iloc[-1]

a, b, c = st.columns(3)

a.metric(
    "RSI (14)",
    f"{last['RSI']:.2f}"
    if pd.notna(last["RSI"]) else "N/A"
)

b.metric(
    "MACD",
    f"{last['MACD']:.3f}"
    if pd.notna(last["MACD"]) else "N/A"
)

c.metric(
    "20-period volatility",
    f"{last['Volatility'] * 100:.2f}%"
    if pd.notna(last["Volatility"]) else "N/A"
)

rsi = last["RSI"]

if pd.isna(rsi):
    st.write("Not enough data to calculate RSI.")
elif rsi >= 70:
    st.write(
        "RSI is relatively high. Some traders describe this as "
        "overbought, but it is not a guaranteed sell signal."
    )
elif rsi <= 30:
    st.write(
        "RSI is relatively low. Some traders describe this as "
        "oversold, but it is not a guaranteed buy signal."
    )
else:
    st.write(
        "RSI is between 30 and 70, a commonly watched range."
    )

# RSI chart
rsi_chart = go.Figure()

rsi_chart.add_trace(
    go.Scatter(
        x=featured.index,
        y=featured["RSI"],
        name="RSI"
    )
)

rsi_chart.add_hline(y=70, line_dash="dash")
rsi_chart.add_hline(y=30, line_dash="dash")

rsi_chart.update_layout(
    title="Relative Strength Index",
    height=300,
    yaxis_range=[0, 100]
)

st.plotly_chart(
    rsi_chart,
    use_container_width=True
)

# MACD chart
macd_chart = go.Figure()

macd_chart.add_trace(
    go.Scatter(
        x=featured.index,
        y=featured["MACD"],
        name="MACD"
    )
)

macd_chart.add_trace(
    go.Scatter(
        x=featured.index,
        y=featured["MACD_Signal"],
        name="Signal"
    )
)

macd_chart.update_layout(
    title="MACD Indicator",
    height=300
)

st.plotly_chart(
    macd_chart,
    use_container_width=True
)

# --------------------------------------------------
# BEGINNER EDUCATION
# --------------------------------------------------

st.header("📚 Learn Stock Market Basics")

with st.expander("What are High and Low?"):
    st.write(
        "High is the highest traded price recorded during a bar. "
        "Low is the lowest traded price recorded during that bar."
    )

with st.expander("What is RSI?"):
    st.write(
        "RSI is a momentum indicator. Values above 70 and below 30 "
        "are commonly watched, but do not guarantee price reversals."
    )

with st.expander("What is MACD?"):
    st.write(
        "MACD compares two exponential moving averages to help "
        "analyze momentum. It can produce false signals."
    )

with st.expander("What is MAE?"):
    st.write(
        "Mean Absolute Error measures the average absolute difference "
        "between predictions and actual prices. Lower is generally "
        "better when comparing the same target and test period."
    )

with st.expander("What is volatility?"):
    st.write(
        "Volatility measures how much prices fluctuate. High volatility "
        "means larger price changes, not necessarily upward movement."
    )

# --------------------------------------------------
# EXPORT DATA
# --------------------------------------------------

st.header("📥 Export Market Data")

st.download_button(
    "Download historical stock data",
    stock.to_csv(),
    file_name=f"{ticker}_historical.csv",
    mime="text/csv"
)

st.caption(
    "Data source: Yahoo Finance through yfinance. "
    "Data coverage and delay may vary by symbol and interval. "
    "Forecasts and indicators are for education and research, "
    "not personalized financial advice."
)
