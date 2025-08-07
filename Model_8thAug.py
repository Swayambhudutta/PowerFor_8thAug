
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.title("Power Forecasting App")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    if "State" not in df.columns:
        st.error("Missing 'State' column in the uploaded file.")
    else:
        state_choice = st.selectbox("Select State", ["Delhi", "Maharashtra"])
        df_state = df[df["State"] == state_choice].copy()

        features = ["temperature_2m (°C)", "rain (mm)", "DNI", "Weekend Tag", "Holiday Tag"]
        target = "Hourly Demand Met (in MW)"

        df_state[features + [target]] = df_state[features + [target]].apply(pd.to_numeric, errors="coerce")
        df_state.dropna(subset=features + [target], inplace=True)

        X = df_state[features]
        y = df_state[target]

        train_size = st.slider("Training Data Percentage", 10, 90, 70)
        split_index = int(len(X) * train_size / 100)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        if len(X_train) == 0 or len(y_train) == 0:
            st.error("Training data is empty after filtering. Please check your input file and training percentage.")
        else:
            model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest"])
            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"R² Score: {r2:.2f}")
