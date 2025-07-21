# weather-prediction-lstm
Real-Time Weather Prediction Dashboard Using LSTM (Kolkata) A Streamlit-based web app that fetches live weather sensor data (temperature, humidity) from ThingSpeak, uses an LSTM model to predict temperature, humidity, wind speed, and rain rate, and displays predictions, latency, and R² scores with auto-refresh and farmer alerts.
# Live Weather Prediction Dashboard 🌤️

This project fetches real-time data from ThingSpeak and predicts weather parameters using an LSTM model.

## 🔧 Features
- Live sensor data from ThingSpeak
- LSTM-based prediction for:
  - Temperature
  - Humidity
  - Rain rate
  - Wind speed
- Real-time auto-refresh
- Latency tracking
- R² evaluation
- Alert system for farmers

## 📁 Files
- `app.py`: Main Streamlit dashboard
- `lstm_model_august_1_to_31.h5`: Trained model
- `august_1_to_31_combined.csv`: Training dataset

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
# Weather Prediction using LSTM

This Streamlit app connects to live weather data from [ThingSpeak](https://thingspeak.com/), uses a pre-trained LSTM model to predict **temperature**, **humidity**, **wind speed**, and **rain rate**. It displays:
- Real-time predictions
- Latency (API to prediction)
- R² scores for accuracy evaluation
- Farmer alerts based on critical weather conditions
- Auto-refresh every 10 seconds

## 💻 Features
- 🌐 Live sensor data
- 📊 Actual vs. Predicted plots
- ⏱️ Latency tracking
- 🚨 Smart alerts for farmers
- 📈 Overall R² evaluation

## 🔧 Setup
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
streamlit run app.py
