
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import joblib
import os

st.title("ZyraPod ANN Simulator for Gasifier")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_gasifier_data.csv")

df = load_data()

# Split data
X = df[['FeedRate', 'Height', 'Diameter', 'Throat', 'ER', 'SBR', 'BD']]
y = df[['CGE', 'SyngasYield', 'Temp_Pyro', 'Temp_Oxid', 'Temp_Reduc',
        'H2', 'CO', 'CO2', 'CH4', 'N2']]

# Normalize inputs
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, "scaler.save")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Build model
model = Sequential([
    Dense(64, input_dim=7, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

# Save model
model.save("ann_gasifier_model.h5")

st.subheader("Enter Gasifier Parameters")

feed_rate = st.slider("Biomass Feed Rate (kg/hr)", 1, 500, 100)
height = st.number_input("Gasifier Height (m)", 0.5, 3.0, 2.0)
diameter = st.number_input("Gasifier Diameter (m)", 0.2, 1.5, 0.8)
throat = st.number_input("Throat Diameter (m)", 0.05, 0.8, 0.2)
er = st.number_input("Equivalence Ratio (ER)", 0.15, 0.35, 0.25)
sbr = st.number_input("Steam to Biomass Ratio (SBR)", 0.0, 1.5, 0.5)
bd = st.number_input("Bulk Density (kg/m³)", 200, 700, 400)

if st.button("Predict"):
    inputs = np.array([[feed_rate, height, diameter, throat, er, sbr, bd]])
    scaler = joblib.load("scaler.save")
    inputs_scaled = scaler.transform(inputs)
    model = load_model("ann_gasifier_model.h5")
    prediction = model.predict(inputs_scaled)[0]

    st.subheader("Prediction Results")
    st.write(f"**CGE:** {prediction[0]:.2f} %")
    st.write(f"**Syngas Yield:** {prediction[1]:.2f} Nm³/kg")
    st.write("**Zone Temperatures (°C):**")
    st.write(f"Pyrolysis: {prediction[2]:.2f}, Oxidation: {prediction[3]:.2f}, Reduction: {prediction[4]:.2f}")
    st.write("**Gas Composition (%):**")
    st.write(f"H₂: {prediction[5]:.2f}, CO: {prediction[6]:.2f}, CO₂: {prediction[7]:.2f}, CH₄: {prediction[8]:.2f}, N₂: {prediction[9]:.2f}")
