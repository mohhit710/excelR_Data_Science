import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app interface
st.title('Titanic Survival Prediction')

# Input fields for user interaction
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=7.25)
pclass = st.selectbox("Pclass", [1, 2, 3])
SibSp = st.number_input("Sibling/spouse" , min_value=0 , max_value =15)
parch = st.number_input("parch" , min_value = 0 , max_value = 15 )

# Convert inputs
sex = 1 if sex == "male" else 0

# Create feature vector from user input
user_input = np.array([[pclass ,sex	,age	,SibSp ,	parch	,fare]])

# Normalize the input using the loaded scaler
user_input_scaled = scaler.transform(user_input)

# Predict the survival using the loaded model
if st.button('Predict Survival'):
    prediction = model.predict(user_input_scaled)
    result = "Survived" if prediction == 1 else "Did not survive"
    st.write(f"The passenger {result}")
