# Diabetes Prediction System 
***
This is a machine learning application built with Streamlit that predicts diabetes risk based on patient information.

## Features

- User-friendly interface for input
- Machine learning-based prediction
- Detailed results with recommendations

## Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the model file and dataset are available:
   - `diabetes.csv` - The dataset
   - `diabetes_model.joblib` - The trained model

   If the model isn't loading correctly, you can regenerate it:
   ```bash
   python save_model.py
   ```

## Running the Application

Launch the Streamlit application:

```bash
streamlit run streamlit.py
```

## Troubleshooting

### ModuleNotFoundError
If you encounter a ModuleNotFoundError:
1. Make sure all dependencies are installed using the requirements.txt file
2. Check if scikit-learn is installed with the correct version
3. If the model won't load, try regenerating it with `save_model.py`

### File Not Found
If files cannot be found:
1. Check the current working directory
2. Make sure all files are in the same directory as the script
3. Use absolute paths if necessary

### Model Incompatibility
If the model is incompatible:
1. Use the compatibility version of the model (`diabetes_model_compat.joblib`)
2. Make sure scikit-learn versions are compatible between training and deployment

## Project Description
This project presents a predictive model for diagnosing diabetes based on the analysis of diagnostic measurements. The dataset utilized is sourced from the National Institute of Diabetes and Digestive and Kidney Diseases, specifically focusing on females aged at least 21 years old of Pima Indian heritage. The primary objective is to predict the likelihood of a patient having diabetes through the examination of various medical predictor variables.
![interface1](https://github.com/MayankTak03/ML-Project-Diabetes-Predication-system/assets/151378644/5a6bae1b-ff96-4f30-b5a5-b3aacf2316e7)

## Objective
The primary goal of this project is to develop an accurate and reliable prediction model that aids in the early diagnosis of diabetes. Leveraging machine learning techniques, the system processes the provided diagnostic measurements to classify patients into diabetic or non-diabetic categories.![img 2](https://github.com/MayankTak03/ML-Project-Diabetes-Predication-system/assets/151378644/676d8ac8-cb94-4142-bbab-3e8d5e7b9983)

## About the Dataset Used
The dataset encompasses critical health metrics, providing a comprehensive view for predictive analysis. The key variables include:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Age
- Outcome: Target variable indicating diabetes presence (1) or absence (0)
![img 1](https://github.com/MayankTak03/ML-Project-Diabetes-Predication-system/assets/151378644/08e15958-8f15-46f5-8ce9-98c4bff4bdee)

## Data Source

The application uses the Pima Indians Diabetes Database. Each record represents a patient and contains various health metrics.

## Model

The prediction model was trained on historical diabetes data. The model file ('diabetes') must be present in the root directory.

## Results
The developed Diabetes Prediction System demonstrates promising accuracy and reliability in identifying individuals at risk of diabetes. The model's performance is assessed through rigorous evaluation metrics, providing insights into its strengths and areas for potential improvement.
![interface1](https://github.com/MayankTak03/ML-Project-Diabetes-Predication-system/assets/151378644/5a6bae1b-ff96-4f30-b5a5-b3aacf2316e7)

## Files Included
Stremlit file:https://github.com/MayankTak03/ML-Project-Diabetes-Predication-system/blob/main/streamlit.py <br>
Frontend part python file:https://github.com/MayankTak03/ML-Project-Diabetes-Predication-system/blob/main/dibetes_main.ipynb <br>
csv File:https://github.com/MayankTak03/ML-Project-Diabetes-Predication-system/blob/main/diabetes.csv <br>

## Project Structure
- `diabetes.csv` - Dataset containing features and target variable
- `streamlit.py` - Web interface for making predictions
- `save_model.py` - Script to train and save the model
- `diabetes_model.joblib` - Trained machine learning model
- `requirements.txt` - Required Python packages
