import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess the data
def load_and_preprocess_data():
    data = pd.read_csv('data.csv')

    # Drop the unnecessary column 'Unnamed: 32'
    if 'Unnamed: 32' in data.columns:
        data = data.drop(columns=['Unnamed: 32'])

    # Drop duplicates if any
    data = data.drop_duplicates()

    # Handle categorical data
    data['diagnosis'] = data['diagnosis'].map({'M': 1 , 'B': 0})

    # Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    data.iloc[:, :] = imputer.fit_transform(data)

    # Check for outliers using Z-score method
    z_scores = np.abs(stats.zscore(data.iloc[:, 2:]))  # Exclude 'id' and 'diagnosis' columns
    outliers = (z_scores > 3).sum(axis=1)
    data = data[(outliers < 1)]  # Keep rows without extreme outliers

    # Normalize data within a specific range
    scaler = StandardScaler()
    data.iloc[:, 2:] = scaler.fit_transform(data.iloc[:, 2:])

    return data, scaler

# Load data and scaler
data, scaler = load_and_preprocess_data()

# Split the data into features and target
X = data.drop(columns=['id', 'diagnosis'])  # Drop 'id' and 'diagnosis' columns
y = data['diagnosis']  # 'diagnosis' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the WMRBF-SVM model
wmrbf_svm_model = SVC(kernel='rbf', class_weight='balanced')
wmrbf_svm_model.fit(X_train, y_train)

# Streamlit app layout
st.title("Breast Cancer Prediction with WMRBF-SVM")

st.write("### Input the features below to predict the diagnosis:")

# Get user input for features
def user_input_features():
    features = {}
    inverse_scaled = pd.DataFrame(scaler.inverse_transform(X_train), columns=X_train.columns)
    for col in X.columns:
        features[col] = st.number_input(col, float(inverse_scaled[col].min()), float(inverse_scaled[col].max()), float(inverse_scaled[col].mean()))
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Display the input dataframe
st.write("### Input Features:")
st.write(input_df)

# Scale the user inputs before prediction
scaled_input = scaler.transform(input_df)

# Predict using the WMRBF-SVM model
if st.button('Predict'):
    prediction = wmrbf_svm_model.predict(scaled_input)
    diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"### Predicted Diagnosis: **{diagnosis}**")

    # Evaluate the model (optional, for display purposes)
    y_pred_svm = wmrbf_svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    st.write(f"WMRBF-SVM Model Accuracy: {accuracy_svm:.4f}")
