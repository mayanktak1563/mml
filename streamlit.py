import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Set page config
st.set_page_config(page_title="Diabetes Prediction", layout="wide", page_icon="🏥")

# Title and description
st.title('Diabetes Prediction using ML')
st.write('Enter the required information to check diabetes risk')

# Try importing required packages
try:
    import joblib
    import sklearn
except ImportError as e:
    st.error(f"Required package not found: {str(e)}")
    st.info("Please install required packages using: pip install -r requirements.txt")
    st.stop()

# Display environment info for debugging
st.sidebar.subheader("Debug Information")
st.sidebar.write(f"Python version: {sys.version}")
st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write(f"Files in directory: {os.listdir('.')}")

# Check if model file exists
model_path = 'diabetes'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please check the file location.")
    st.info(f"Current working directory: {os.getcwd()}")
    st.stop()

# Load model with error handling
@st.cache_resource
def load_model(model_path):
    try:
        st.info(f"Attempting to load model from: {model_path}")
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("This could be due to missing dependencies or incompatible model version.")
        st.info("Make sure scikit-learn is installed with: pip install scikit-learn")
        return None

# Load data with error handling
@st.cache_data
def load_data():
    try:
        data_path = 'diabetes.csv'
        if not os.path.exists(data_path):
            st.warning(f"Data file '{data_path}' not found. Using example data instead.")
            # Create sample data
            columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            sample_data = pd.DataFrame({
                'Pregnancies': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                'Glucose': list(range(70, 200, 10)),
                'BloodPressure': list(range(40, 130, 10)),
                'SkinThickness': list(range(0, 100, 10)),
                'Insulin': list(range(0, 850, 100)),
                'BMI': [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
                'DiabetesPedigreeFunction': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0],
                'Age': list(range(21, 81, 5))
            })
            return sample_data
        return pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load model and data
diabetes_model = load_model(model_path)
df = load_data()

if df.empty:
    st.error("Could not load the dataset. Please check if the diabetes.csv file is available.")
    st.stop()

if diabetes_model is None:
    # Create a fallback section to still let users see the form but without predictions
    st.warning("Model could not be loaded. You can view the form but predictions will not work.")
    show_predictions = False
else:
    show_predictions = True

# Create form for better submission control
with st.form("prediction_form"):
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
    with col2:
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=85)
    with col3:
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=130, value=66)

    with col1:
        SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=29)
    with col2:
        Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=846, value=0)
    with col3:
        BMI = st.number_input('BMI value (kg/m²)', min_value=0.0, max_value=70.0, value=26.6, step=0.1)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.351, step=0.001)
    with col2:
        Age = st.number_input('Age (years)', min_value=21, max_value=81, value=31)
    
    # Create a submit button
    submit_button = st.form_submit_button(label="Predict Diabetes Risk")

# Prediction processing
if submit_button:
    try:
        if not show_predictions:
            st.error("Cannot make predictions because the model failed to load.")
        else:
            # Convert input values to appropriate data types
            input_data = np.array([
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]).reshape(1, -1)
            
            # Make prediction
            diab_prediction = diabetes_model.predict(input_data)
            
            # Display result
            st.subheader("Prediction Result")
            if diab_prediction[0] == 1:
                st.error("🔴 The person is diabetic")
                st.info("Please consult with a healthcare professional for proper diagnosis and treatment.")
            else:
                st.success("🟢 The person is not diabetic")
                st.info("Remember to maintain a healthy lifestyle for diabetes prevention.")
                
            # Display input data summary
            st.subheader("Patient Data Summary")
            patient_data = {
                'Parameter': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                            'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'],
                'Value': [Pregnancies, Glucose, BloodPressure, SkinThickness, 
                        Insulin, BMI, DiabetesPedigreeFunction, Age]
            }
            st.table(pd.DataFrame(patient_data))
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please check your input values and try again.")

# Add instructions for deploying
st.sidebar.subheader("Deployment Instructions")
st.sidebar.markdown("""
1. Make sure all required packages are installed:
   ```
   pip install -r requirements.txt
   ```
2. Ensure the model file and dataset are in the same directory as this script
3. Run the app with:
   ```
   streamlit run streamlit.py
   ```
""")
