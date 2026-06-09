import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Page Configuration
st.set_page_config(page_title="Isolation Forest", layout="wide")

st.title("🔍 Anomaly Detection using Isolation Forest")

# Load Dataset
df = pd.read_csv("data/Mall_Customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Select numerical columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Remove CustomerID if present
if "CustomerID" in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=["CustomerID"])

st.subheader("Features Used")
st.write(numeric_df.columns.tolist())

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# Model Parameters
st.subheader("Isolation Forest Parameters")

contamination = st.slider(
    "Contamination Ratio",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01
)

# Train Model
model = IsolationForest(
    contamination=contamination,
    random_state=42
)

predictions = model.fit_predict(X_scaled)

# Add Predictions to DataFrame
df["Anomaly"] = predictions

# Convert Labels
df["Anomaly Label"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# Metrics
num_anomalies = (df["Anomaly"] == -1).sum()
num_normal = (df["Anomaly"] == 1).sum()

col1, col2 = st.columns(2)

with col1:
    st.metric("Normal Points", num_normal)

with col2:
    st.metric("Anomalies Detected", num_anomalies)

# Visualization
st.subheader("Anomaly Visualization")

fig, ax = plt.subplots(figsize=(8, 6))

normal_data = df[df["Anomaly"] == 1]
anomaly_data = df[df["Anomaly"] == -1]

ax.scatter(
    normal_data["Annual Income (k$)"],
    normal_data["Spending Score (1-100)"],
    label="Normal",
    alpha=0.7
)

ax.scatter(
    anomaly_data["Annual Income (k$)"],
    anomaly_data["Spending Score (1-100)"],
    label="Anomaly",
    marker="x",
    s=100
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Isolation Forest Anomaly Detection")
ax.legend()

st.pyplot(fig)

# Full Dataset
st.subheader("Dataset with Predictions")
st.dataframe(df)

# Anomalies Only
st.subheader("Detected Anomalies")
st.dataframe(df[df["Anomaly"] == -1])

st.success("Isolation Forest Completed Successfully!")