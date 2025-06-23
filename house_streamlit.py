# house_price_app.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.stats.stattools import durbin_watson
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Streamlit page config
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction using Linear Regression")
st.subheader("Welcome 👋")

# File Upload
file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.subheader("📋 Data Preview")
    st.dataframe(df)

    # Feature selection
    X = df[["Area", "Bedrooms", "Bathrooms", "Location_Score", "Age"]]
    y = df["Price"]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Evaluation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    residuals = y_test - y_pred
    dw = durbin_watson(residuals)

    st.subheader("📈 Model Evaluation")
    st.metric("R² Score", f"{r2:.2f}")
    st.metric("MAE", f"₹{mae:,.2f}")
    st.metric("MSE", f"₹{mse:,.2f}")
    st.metric("Durbin-Watson", f"{dw:.2f}")

    # Coefficients
    coef_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
    st.subheader("📌 Feature Importance")
    st.dataframe(coef_df)

    # Prediction input
    st.subheader("🔍 Predict New House Price")
    area = st.number_input("Area (sq.ft)", 300, 5000, 1000, step=50)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 6, 2)
    age = st.slider("Age of House (years)", 0, 50, 5)
    loc_score = st.slider("Location Score (0-10)", 0.0, 10.0, 7.5, step=0.1)

    input_data = np.array([[area, bedrooms, bathrooms, loc_score, age]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"💰 Predicted Price: ₹{predicted_price:,.2f}")

    # Actual vs Predicted
    st.subheader("📉 Actual vs Predicted")
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions', marker=dict(color='blue')))
    scatter_fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                     mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))
    scatter_fig.update_layout(title="Actual vs Predicted House Prices",
                              xaxis_title="Actual Price", yaxis_title="Predicted Price")
    st.plotly_chart(scatter_fig)

    # Residual Plot
    st.subheader("📉 Residual Plot")
    residual_fig = px.scatter(x=y_pred, y=residuals,
                              labels={'x': 'Predicted Price', 'y': 'Residuals'},
                              title="Residuals vs Predicted Price")
    residual_fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(residual_fig)

    # Download results
    result_df = pd.DataFrame({"Actual Price": y_test, "Predicted Price": y_pred})
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')

else:
    st.info("Upload a CSV file with columns: Area, Bedrooms, Bathrooms, Location_Score, Age, and Price")
