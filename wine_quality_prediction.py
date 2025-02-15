import numpy as np
from PIL import Image
import streamlit as st
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open('wine_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Predict car price
def predict_car_price(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction


# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="wide")


    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stNumberInput>div>div>input {
                font-size: 16px;
            }
            .stSelectbox>div>div>select {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🍷 Wine Quality Prediction")
    st.markdown("This app predicts the quality of a wine based on its specifications.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        try:
            profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
            st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")
        except:
            st.warning("Profile image not found.")

        st.title("About")
        st.info("This app uses a machine learning model to predict the quality of a wine")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, value=7.0, help="Enter the fixed acidity of the wine")
        volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.3, help="Enter the volatile acidity of the wine")
        citric_acid = st.number_input("Citric Acid", min_value=0.0, value=0.3, help="Enter the citric acid content of the wine")
        residual_sugar = st.number_input("Residual Sugar", min_value=0.0, value=20.0, help="Enter the residual sugar content of the wine")
        chlorides = st.number_input("Chlorides", min_value=0.0, value=0.08, help="Enter the chloride content of the wine") 
        

    with col2:
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, value=15.0, help="Enter the free sulfur dioxide content of the wine")
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, value=70.0, help="Enter the total sulfur dioxide content of the wine")
        density = st.number_input("Density", min_value=0.0, value=1.0, help="Enter the density of the wine")
        pH = st.number_input("pH", min_value=0.0, value=3.0, help="Enter the pH level of the wine")
        sulphates = st.number_input("Sulphates", min_value=0.0, value=0.5, help="Enter the sulphates content of the wine")

    alcohol = st.number_input("Alcohol", min_value=0.0, value=9.0, help="Enter the alcohol content of the wine")
    
    # Prepare input data for the model
    input_data = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]

    # Prediction button
    if st.button("Predict"):
        try:
            prediction = predict_car_price(input_data)

            if (prediction[0]==1):
                prediction = 'Good Quality Wine'
                st.success(f"**{prediction}**")
                result_placeholder.success(f"**{prediction}**")
            else:
                prediction = 'Bad Quality Wine'
                st.error(f"**{prediction}**")
                result_placeholder.error(f"**{prediction}**")

            

        except Exception as e:
            st.error(f"An error occurred: {e}")
            result_placeholder.error("An error occurred during prediction. Please check the input data.")

if __name__ == "__main__":
    main()

# Run the app: streamlit run wine_quality_prediction.py