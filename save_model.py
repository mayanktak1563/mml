# This script re-saves the model in a compatible format
import joblib
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

# Display sklearn version for reference
import sklearn
print(f"Using scikit-learn version: {sklearn.__version__}")

# Load data
print("Loading data...")
try:
    diabetes_dataset = pd.read_csv('diabetes.csv')
    print(f"Data loaded successfully with shape: {diabetes_dataset.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Prepare data
print("Preparing data...")
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
print("Training model...")
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Save model
print("Saving model...")
try:
    joblib.dump(classifier, 'diabetes_model.joblib', compress=3)
    print("Model saved as 'diabetes_model.joblib'")
    
    # Also save with protocol=2 for better compatibility
    joblib.dump(classifier, 'diabetes_model_compat.joblib', compress=3, protocol=2)
    print("Compatibility model saved as 'diabetes_model_compat.joblib'")
except Exception as e:
    print(f"Error saving model: {e}")

print("Done!")
