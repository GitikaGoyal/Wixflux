# 📈 Stock Price Prediction using LSTM

This project predicts future stock prices using LSTM (Long Short-Term Memory) deep learning models. It uses historical data of Apple Inc. (AAPL) downloaded via Yahoo Finance and provides a 30-day forecast.

---

## 🔍 Problem Statement
Predict future stock closing prices based on past 60-day windows of data.

## 🧠 Model Used
- Long Short-Term Memory (LSTM)
- Sequential Keras Model

## 🗃️ Dataset
- Source: [Yahoo Finance](https://finance.yahoo.com)
- Ticker: AAPL
- Date Range: 2010 to Present

## 📊 Evaluation
- Visual evaluation (forecast vs actual prices)
- Mean Squared Error (loss)

## 🔧 Requirements

```bash
pip install tensorflow yfinance scikit-learn pandas matplotlib
