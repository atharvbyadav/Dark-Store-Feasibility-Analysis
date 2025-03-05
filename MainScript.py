import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def generate_dummy_data():
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    demand = np.random.randint(50, 500, size=len(dates)) + np.sin(np.linspace(0, 10, len(dates))) * 50
    localities = np.random.choice(['Hinjewadi', 'Baner', 'Viman Nagar', 'Kothrud', 'Magarpatta'], len(dates))
    return pd.DataFrame({'Date': dates, 'Locality': localities, 'Demand': demand})

data = generate_dummy_data()
data['DayOfYear'] = data['Date'].dt.dayofyear

data = pd.get_dummies(data, columns=['Locality'], drop_first=True)
data.set_index('Date', inplace=True)

X = data.drop(columns=['Demand'])
y = data['Demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Streamlit App UI
st.title("📈 Demand Forecasting Dashboard")
st.markdown("### Predict future demand for different localities")

# Show Data
if st.checkbox("Show Raw Data"):
    st.write(data.head())

# Plot Actual vs Predicted Demand
fig = px.line(x=X_test.index, y=[y_test, y_pred], labels={'x': "Date", 'y': "Demand"},
              title="Actual vs Predicted Demand")
fig.update_traces(mode="markers+lines")
st.plotly_chart(fig)

# Model Evaluation Metrics
st.subheader("Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Future Predictions
st.subheader("Predict Future Demand")
days_ahead = st.slider("Select number of days ahead for prediction:", 1, 30, 7)

# Create Future Dates
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_ahead, freq='D')
future_days = np.arange(data['DayOfYear'].max()+1, data['DayOfYear'].max()+1+days_ahead)

# Create Future DataFrames
future_X = pd.DataFrame({'DayOfYear': future_days})

# Ensure future_X has the same dummy variable structure
for col in X_train.columns:
    if col not in future_X.columns:
        future_X[col] = 0  # Add missing locality columns with default 0

# Predict future demand
future_pred = model.predict(future_X)

# Create DataFrame for visualization
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Demand': future_pred})
st.write(future_df)

# Plot predictions
fig_future = px.line(future_df, x='Date', y='Predicted Demand', title="Future Demand Prediction")
st.plotly_chart(fig_future)
