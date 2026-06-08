import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Live Stock Prediction App")
st.markdown("**Real-time Training + Accuracy**")

feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA14', 'MA30', 
                   'RSI', 'MACD', 'BB_Upper', 'BB_Lower']

def train_and_predict(symbol):
    with st.spinner(f"Downloading data & training model for {symbol}..."):
        
        df = yf.download(symbol, start='2023-01-01', end='2025-06-07', progress=False)
        df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        if len(df) < 100:   # Minimum required data
            st.error(f"❌ Not enough historical data for {symbol}. This stock is very new or has limited trading history.")
            st.info("Please try a well-established stock like RELIANCE.NS, TCS.NS, HDFCBANK.NS etc.")
            return None

        # Technical Indicators
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA14'] = df['Close'].rolling(window=14).mean()
        df['MA30'] = df['Close'].rolling(window=30).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']

        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

        # Target
        future_days = 5
        df['Future_Close'] = df['Close'].shift(-future_days)
        df['Target'] = np.where(df['Future_Close'] > df['Close'], 1, 0)

        df = df.dropna()

        if len(df) < 50:
            st.warning("⚠️ Very less data left after processing. Prediction may not be reliable.")

        X = df[feature_columns]
        y = df['Target']

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                            random_state=42, shuffle=False)

        # Train Model
        model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
        model.fit(X_train, y_train)

        # Calculate Accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100

        # Current Prediction
        latest = X_scaled[-1:].reshape(1, -1)
        prediction = model.predict(latest)[0]
        probability = model.predict_proba(latest)[0]

        current_price = df['Close'].iloc[-1]

        # Expected Price
        expected_change = 0.018 * probability[1] if prediction == 1 else -0.012 * probability[0]
        expected_price = current_price * (1 + expected_change)

        return {
            'current_price': current_price,
            'signal': '🟢 BUY' if prediction == 1 else '🔴 SELL / HOLD',
            'buy_prob': probability[1]*100,
            'accuracy': accuracy,
            'expected_price': expected_price,
            'expected_change': expected_change*100
        }

# ================== UI ==================
symbol = st.text_input("Enter Stock Symbol (e.g. RELIANCE.NS, PFC.NS, TCS.NS)", 
                       "RELIANCE.NS").upper().strip()

if st.button("Train Model & Get Prediction"):
    if not symbol.endswith('.NS'):
        symbol = symbol + ".NS"
    
    result = train_and_predict(symbol)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"₹{result['current_price']:.2f}")
    with col2:
        st.metric("Signal", result['signal'])
    with col3:
        st.metric("Expected in 5 Days", f"₹{result['expected_price']:.2f}")

    st.progress(result['buy_prob']/100)
    st.success(f"**Model Accuracy: {result['accuracy']:.2f}%**")
    st.write(f"**Buy Probability: {result['buy_prob']:.2f}%**")
    st.info(f"**Expected Change: {result['expected_change']:+.2f}%**")

st.caption("Note: Model is trained fresh every time. Accuracy is on test data.")
