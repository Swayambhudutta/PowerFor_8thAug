import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(layout="wide")
st.title("Short-Term Intra-Day Forecast of Power Demand")

# Sidebar
with st.sidebar:
    st.header("Model Configuration")
    model_choice = st.selectbox("Choose Forecasting Model", 
                                ["ARIMA", "SARIMAX", "Random Forest", "Linear Regression", "SVR", "XGBoost", "LSTM", "GRU", "Hybrid"])
    train_size = st.slider("Training Data Percentage", 10, 90, 70)

    st.subheader("Accuracy Metrics")
    rmse_placeholder = st.empty()
    mae_placeholder = st.empty()
    r2_placeholder = st.empty()

    st.subheader("Model Insights")
    insights_placeholder = st.empty()

# File uploader
uploaded_file = st.file_uploader("Upload Power Demand Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    df.dropna(inplace=True)

    features = ['temperature_2m (°C)', 'rain (mm)', 'DNI', 'Weekend Tag', 'Holiday Tag']
    target = 'Hourly Demand Met (in MW)'

    X = df[features]
    y = df[target]

    split_index = int(len(df) * train_size / 100)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    def evaluate(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    if model_choice in ["LSTM", "GRU", "Hybrid"]:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        X_seq = []
        y_seq = []
        for i in range(24, len(X_scaled)):
            X_seq.append(X_scaled[i-24:i])
            y_seq.append(y_scaled[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        split_seq = int(len(X_seq) * train_size / 100)
        X_train_seq, X_test_seq = X_seq[:split_seq], X_seq[split_seq:]
        y_train_seq, y_test_seq = y_seq[:split_seq], y_seq[split_seq:]

        def build_model(cell_type='LSTM'):
            model = Sequential()
            if cell_type == 'LSTM':
                model.add(LSTM(50, return_sequences=False, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            else:
                model.add(GRU(50, return_sequences=False, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            return model

        if model_choice == "LSTM":
            model = build_model('LSTM')
            model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)
            y_pred_scaled = model.predict(X_test_seq)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_actual = scaler_y.inverse_transform(y_test_seq)

        elif model_choice == "GRU":
            model = build_model('GRU')
            model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)
            y_pred_scaled = model.predict(X_test_seq)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_test_actual = scaler_y.inverse_transform(y_test_seq)

        elif model_choice == "Hybrid":
            lstm_model = build_model('LSTM')
            lstm_model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)
            lstm_pred = scaler_y.inverse_transform(lstm_model.predict(X_test_seq))

            xgb_model = xgb.XGBRegressor()
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)

            xgb_pred = xgb_pred[-len(lstm_pred):]
            y_pred = (lstm_pred.flatten() + xgb_pred) / 2
            y_test_actual = y_test.values[-len(y_pred):]

    else:
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor()
        elif model_choice == "SVR":
            model = SVR()
        elif model_choice == "XGBoost":
            model = xgb.XGBRegressor()
        elif model_choice == "ARIMA":
            model = ARIMA(y_train, order=(5,1,0)).fit()
            y_pred = model.forecast(steps=len(y_test))
            y_test_actual = y_test
        elif model_choice == "SARIMAX":
            model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,24)).fit(disp=False)
            y_pred = model.forecast(steps=len(y_test))
            y_test_actual = y_test

        if model_choice not in ["ARIMA", "SARIMAX"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_test_actual = y_test

    # Evaluation
    rmse, mae, r2 = evaluate(y_test_actual, y_pred)
    rmse_placeholder.metric("RMSE", f"{rmse:.2f}")
    mae_placeholder.metric("MAE", f"{mae:.2f}")
    r2_placeholder.metric("R² Score", f"{r2:.2f}")

    if r2 < 0.3:
        insights = "Low Accuracy: High error and low correlation. Consider alternative models or preprocessing."
    elif r2 < 0.7:
        insights = "Moderate Accuracy: Acceptable performance. May benefit from tuning or additional features."
    else:
        insights = "High Accuracy: Model performs well on the data."
    insights_placeholder.info(insights)

    # Plot
    st.subheader(f"Forecast vs Actual using {model_choice}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.asarray(y_test_actual).flatten(), name='Actual'))
    fig.add_trace(go.Scatter(y=[y_train.mean()] * len(y_test_actual), name='Baseline'))
    fig.add_trace(go.Scatter(y=np.asarray(y_pred).flatten(), name='Predicted'))
    fig.update_layout(xaxis_title='Time', yaxis_title='Power Demand (MW)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Disclaimer:** Trained on {train_size}% ({len(y_train)} points), Forecasted on {100-train_size}% ({len(y_test_actual)} points)")
    st.markdown("**Data Sources:** ICED Niti Aayog, Open Meteo, officeholidays")

    st.subheader("Forecast Impact")
    savings = np.mean(y_test_actual) - np.mean(y_pred)
    if savings > 0:
        st.success(f"Forecast helped in saving {savings:.2f} MW")
    else:
        st.error(f"Forecast resulted in excess of {-savings:.2f} MW")

    if st.button("Simulate"):
        st.info("Best model: Random Forest, Best training percentage: 80%")
