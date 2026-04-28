import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('iris_model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Iris Classifier", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌸 Iris Flower Classification App")
st.write("Predict the species of Iris flower based on input features")

st.divider()

# Input section
st.subheader("📥 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
    sepal_width = st.number_input("Sepal Width", 2.0, 4.5, 3.5)

with col2:
    petal_length = st.number_input("Petal Length", 1.0, 7.0, 1.4)
    petal_width = st.number_input("Petal Width", 0.1, 2.5, 0.2)

# Prediction
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = ["Setosa 🌿", "Versicolor 🌼", "Virginica 🌺"]

    st.success(f"Prediction: **{species[prediction[0]]}**")

st.divider()

# Footer
st.caption("Built using Streamlit | ML Model: KNN")