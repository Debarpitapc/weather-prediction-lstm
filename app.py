import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pytz
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load the trained model
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model_august_1_to_31.h5", compile=False)

model = load_lstm_model()

# Dummy values for unused inputs
DUMMY_PM25 = 0.0
DUMMY_WIND = 0.0
DUMMY_RAIN = 0.0

# ThingSpeak API and timezone
url = "https://api.thingspeak.com/channels/3007544/feeds.json"
ist = pytz.timezone("Asia/Kolkata")

# Page config and auto-refresh
st.set_page_config(page_title="Live Weather Output - Kolkata", layout="centered")
st_autorefresh(interval=10 * 1000, limit=None, key="auto_refresh")

# Title
st.title("Live Weather Output - Kolkata")
st.write("This dashboard fetches real-time sensor values and predicts temperature and humidity using an LSTM model.")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

try:
    start_time = datetime.datetime.now()

    # Fetch live sensor data
    response = requests.get(url)
    data = response.json()['feeds']
    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert(ist)
    df = df.rename(columns={'field1': 'Temperature', 'field2': 'Humidity'})
    df[['Temperature', 'Humidity']] = df[['Temperature', 'Humidity']].astype(float)
    latest = df.tail(1).reset_index(drop=True)
    timestamp = df['created_at'].iloc[-1]

    temp_actual = round(latest['Temperature'][0], 2)
    hum_actual = round(latest['Humidity'][0], 2)

    # Prepare input for model
    model_input = [DUMMY_PM25, temp_actual, hum_actual, DUMMY_WIND, DUMMY_RAIN]
    scaler = MinMaxScaler()
    scaler.fit(np.array([model_input]))
    scaled_input = scaler.transform(np.array([model_input])).reshape(1, 1, 5)

    # Prediction
    prediction = model.predict(scaled_input, verbose=0)
    output = scaler.inverse_transform(prediction.reshape(1, -1))[0]
    temp_output = round(output[1], 2)
    hum_output = round(output[2], 2)

    # Latency
    latency = round((datetime.datetime.now() - start_time).total_seconds(), 2)

    # Store results in session
    st.session_state.history.append({
        'Timestamp': timestamp,
        'Actual Temp (°C)': temp_actual,
        'Predicted Temp (°C)': temp_output,
        'Actual Humidity (%)': hum_actual,
        'Predicted Humidity (%)': hum_output,
        'Latency (s)': latency
    })

    # Display predictions
    col1, col2 = st.columns(2)
    col1.metric("Temperature (°C)", f"{temp_output:.2f}")
    col2.metric("Humidity (%)", f"{hum_output:.2f}")
    st.caption(f"Last synced: {timestamp}")

    # CONDITIONAL RECOMMENDATION
    if temp_output > 28 and hum_output > 80:
        st.subheader("🌾 Urgent Alert for Farmers")
        st.error(
            "⚠️ High humidity and temperature detected!\n\n"
            "Current Conditions: Temperature > 28°C and Humidity > 80%\n\n"
            "🌿 These conditions may increase the risk of fungal diseases in crops.\n"
            "🔧 Please consider preventive measures such as:\n"
            "- Spraying appropriate fungicides\n"
            "- Improving field ventilation\n"
            "- Avoiding excess irrigation during this period."
        )

except Exception as e:
    st.error("Failed to retrieve or process data.")
    st.exception(e)

# Plotting and R² display
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history).set_index("Timestamp")

    st.download_button("📥 Download CSV", history_df.to_csv().encode('utf-8'), file_name="weather_predictions_kolkata.csv", mime='text/csv')

    # Plot: Temperature
    st.subheader("🌡️ Temperature Prediction")
    fig_temp, ax_temp = plt.subplots(figsize=(10, 4))
    ax_temp.plot(history_df.index, history_df["Actual Temp (°C)"], label="Actual Temp", color="tomato", linewidth=2)
    ax_temp.plot(history_df.index, history_df["Predicted Temp (°C)"], '--', label="Predicted Temp", color="orange", linewidth=2)
    ax_temp.set_title("Actual vs Predicted Temperature")
    ax_temp.set_xlabel("Time")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.tick_params(axis='x', rotation=45)
    ax_temp.grid(True)
    ax_temp.legend()
    st.pyplot(fig_temp)

    # Plot: Humidity
    st.subheader("💧 Humidity Prediction")
    fig_hum, ax_hum = plt.subplots(figsize=(10, 4))
    ax_hum.plot(history_df.index, history_df["Actual Humidity (%)"], label="Actual Humidity", color="deepskyblue", linewidth=2)
    ax_hum.plot(history_df.index, history_df["Predicted Humidity (%)"], '--', label="Predicted Humidity", color="blue", linewidth=2)
    ax_hum.set_title("Actual vs Predicted Humidity")
    ax_hum.set_xlabel("Time")
    ax_hum.set_ylabel("Humidity (%)")
    ax_hum.tick_params(axis='x', rotation=45)
    ax_hum.grid(True)
    ax_hum.legend()
    st.pyplot(fig_hum)

    # Plot: Latency
    st.subheader("⏱️ Latency Over Time")
    fig_lat, ax_lat = plt.subplots(figsize=(10, 3))
    ax_lat.plot(history_df.index, history_df["Latency (s)"], label="Latency", color="purple", linewidth=2)
    ax_lat.set_title("API-to-Prediction Latency")
    ax_lat.set_xlabel("Time")
    ax_lat.set_ylabel("Seconds")
    ax_lat.tick_params(axis='x', rotation=45)
    ax_lat.grid(True)
    ax_lat.legend()
    st.pyplot(fig_lat)

    
else:
    st.info("Waiting for live data to populate. The dashboard auto-refreshes every 10 seconds.")
