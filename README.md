# Smart Stock Movement Predictor

An end-to-end Machine Learning web application that predicts **short-term stock movement** (Up or Down) for any Indian stock using Technical Analysis and Random Forest algorithm.
 
<img width="1920" height="906" alt="image" src="https://github.com/user-attachments/assets/a9d64d04-6f0d-4f33-a568-20f816397f89" /> <img width="1909" height="906" alt="image" src="https://github.com/user-attachments/assets/bb193178-5d47-4bf6-9740-705c265f7854" /> <img width="1920" height="902" alt="image" src="https://github.com/user-attachments/assets/8344d3ae-fb6b-425b-a29a-ba241f2c86e6" />




## 🎯 Features
- Real-time stock data fetching from Yahoo Finance
- 12+ Technical Indicators (MA, RSI, MACD, Bollinger Bands etc.)
- Machine Learning Model (Random Forest Classifier)
- Live Web Dashboard using Streamlit
- Model trains on selected stock in real-time
- Shows Accuracy, Buy Probability & Expected Price
- Proper error handling for new stocks

## 🛠️ Technologies Used
- **Python**
- **Pandas & NumPy** - Data Processing
- **yfinance** - Stock Data
- **Scikit-learn** - Machine Learning
- **Streamlit** - Web Interface
- Technical Analysis

## 📊 Model Performance
- Average Accuracy: **57% - 62%**
- Best Accuracy observed: **~64%** (on some stocks)
- Predicts next **5 trading days** direction

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/stock-predictor.git

# 2. Go to project folder
cd stock-predictor

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
