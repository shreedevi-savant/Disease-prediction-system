import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# 1. Load Dataset
data = pd.read_csv('dataset.csv')

# Features and Target variable
X = data[['Fever', 'Cough', 'Headache', 'Fatigue']]
y = data['Disease']

# 2. Train Random Forest Model
model = RandomForestClassifier()
model.fit(X, y)

# 3. Streamlit User Interface
st.title("🩺 AI Disease Prediction System")
st.write("Select your symptoms below:")

fever = st.checkbox("Fever")
cough = st.checkbox("Cough")
headache = st.checkbox("Headache")
fatigue = st.checkbox("Fatigue")

if st.button("Predict Disease"):
    input_data = [[int(fever), int(cough), int(headache), int(fatigue)]]
    prediction = model.predict(input_data)
    st.success(f"Predicted Disease: *{prediction[0]}*")