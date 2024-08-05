import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model from the pickle file
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Function to make predictions based on predict_proba and user-defined percentage
def predict_top_percent(input_data, percent):
    proba = model.predict_proba(input_data)
    
    # Assuming binary classification, pick the probability of the positive class
    # For multi-class, modify this accordingly
    proba_positive = proba[:, 1]
    threshold = np.percentile(proba_positive, 100 - percent)  # Calculate the threshold for the top X%
    
    top_indices = np.where(proba_positive >= threshold)[0]
    top_predictions = input_data.iloc[top_indices].copy()
    top_predictions['Prediction'] = model.classes_[1]  # Assuming binary classification
    
    return top_predictions

# Streamlit interface
st.title("Data Prediction Chatbot")

# Upload a CSV file for prediction
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    input_data = pd.read_csv(uploaded_file)
    input_data = input_data.drop(columns='Unnamed: 0')
    st.write("Input Data:")
    st.write(input_data)

    # Prompt for percentage input
    st.write("Enter the percentage of results you want to display (e.g., 30 for top 30%):")
    percentage_input = st.chat_input()

    if percentage_input:
        try:
            percent = float(percentage_input)
            if percent < 0 or percent > 100:
                st.error("Please enter a percentage between 0 and 100.")
            else:
                # Make predictions and get the top X% results
                top_predictions = predict_top_percent(input_data, percent)
                
                # Show the top X% results
                st.write(f"Top {percent}% Predictions:")
                st.write(top_predictions)
                
                # Download the top X% predictions as a CSV file
                csv = top_predictions.to_csv(index=False).encode('utf-8')
                st.download_button(label=f"Download Top {int(percent)}% Predictions as CSV",
                                   data=csv,
                                   file_name=f'top_{int(percent)}_predictions.csv',
                                   mime='text/csv')
        except ValueError:
            st.error("Invalid input. Please enter a numeric value.")
