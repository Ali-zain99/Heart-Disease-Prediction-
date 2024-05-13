import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer

st.sidebar.header('User input Features')

def user_input_features():
    age = st.number_input("Enter your age: ", min_value=20, max_value=100, key="age")
    prevalentHyp = st.sidebar.selectbox('Prevalent Hypertension', (0, 1), key="prevalentHyp")
    prevalentStroke = st.sidebar.selectbox('Prevalent Stroke', (0, 1), key="prevalentStroke")
    diabetes = st.sidebar.selectbox('Diabetes', (0, 1), key="diabetes")
    totChol = st.sidebar.number_input("Total Cholesterol", min_value=100, max_value=600, key="totChol")
    sysBP = st.sidebar.number_input("Systolic Blood Pressure", min_value=80, max_value=300, key="sysBp")
    diaBP = st.sidebar.number_input("Diastolic Blood Pressure", min_value=40, max_value=200, key="diaBp")
    BMI = st.sidebar.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, key="BMI")
    heartRate = st.sidebar.number_input("Heart Rate", min_value=30, max_value=150, key="heartRate")
    glucose = st.sidebar.number_input("Glucose Level", min_value=40, max_value=500, key="glucose")
    male = st.sidebar.selectbox('Gender', ('Female', 'Male'), key="male")
    male = 1 if male == 'Male' else 0
    
    data = {'male': male, 'age': age, 'prevalentStroke': prevalentStroke, 'prevalentHyp': prevalentHyp,
            'diabetes': diabetes, 'totChol': totChol, 'sysBP': sysBP, 'diaBP': diaBP, 'BMI': BMI,
            'heartRate': heartRate, 'glucose': glucose}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Load the dataset
heart_dataset = pd.read_csv('framingham.csv')
heart_dataset = heart_dataset.drop(columns=['education', 'currentSmoker'])

# Concatenate input with dataset
df = pd.concat([input_df, heart_dataset], axis=0)

# One-hot encode categorical variables
# df = pd.get_dummies(df, columns=['male', 'prevalentStroke', 'prevalentHyp', 'diabetes'])

# Take only the first row (user input)
df=df[:1]
df=df.drop([ 'cigsPerDay','BPMeds'], axis=1)
imputer = SimpleImputer(strategy='mean')
imputer.fit(df)
df_imputed = imputer.transform(df)
# Load the trained model
load_clf = pickle.load(open('logistic regression model.pkl', 'rb'))

# Predict TenYearCHD
prediction = load_clf.predict(df_imputed)
prediction_proba = load_clf.predict_proba(df_imputed)

st.title('Heart Disease Prediction System')
st.subheader('Prediction for TenYearCHD')
if prediction[0] == 0:
    st.write("No risk of coronary heart disease in the next 10 years.")
else:
    st.write("You have a Risk of coronary heart disease in the next 10 years.")
