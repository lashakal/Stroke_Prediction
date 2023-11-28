import streamlit as st
import pandas as pd

st.title('Stroke Prediction  :heart:')
st.caption('Created by:   Amadin Ahmed   Lasha Kaliashvili   Mikheil Uglava')
st.text("")
st.markdown("According to the World Health Organization (WHO), stroke is the leading cause of disability worldwide and the second leading cause of death. Over the last 17years, the lifetime risk of developing a stroke has increased by 50% and now 1 in 4 people is estimated to have a stroke in their lifetime. In this project, we are aiming to design a stroke prediction model with a user-friendly interface that can assess an individual’s risk of stroke based on their health data and medical history using the SVM classification algorithm. ")
st.text("")



# Define a function to make predictions based on heart data
def predict_heart_risk(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    # Define the column names based on the heart data features
    columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    
    # Create a DataFrame for the input data
    data = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]], columns=columns)
    
    # Use the model to make a prediction
    prediction = model.predict(data)[0]
    return prediction, data.iloc[0].tolist()

# Helper function to convert numerical data to human-readable form
def display_readable_form(user_input):
    gender_text = "Female" if user_input[0] == 0 else "Male"
    hypertension_text = "No" if user_input[2] == 0 else "Yes"
    heart_disease_text = "No" if user_input[3] == 0 else "Yes"
    ever_married_text = "No" if user_input[4] == 0 else "Yes"
    work_type_text = ["Govt_job", "Never_worked", "Private", "Self_employed", "Children"][user_input[5]]
    Residence_type_text = "Rural" if user_input[6] == 0 else "Urban"
    smoking_status_text = ["Unknown", "Formerly Smoked", "Never Smoked", "Smokes"][user_input[9]]
    
    return f"Gender: {gender_text}, Age: {user_input[1]}, Hypertension: {hypertension_text}, Heart Disease: {heart_disease_text}, Ever Married: {ever_married_text}, Work Type: {work_type_text}, Residence Type: {Residence_type_text}, Avg Glucose Level: {user_input[7]}, BMI: {user_input[8]}, Smoking Status: {smoking_status_text}"

# User interface in Streamlit to collect user inputs
gender = st.radio("Gender", (0, 1))  # 0: Female, 1: Male
age = st.number_input("Age", min_value=0)
hypertension = st.radio("Hypertension", (0, 1))  # 0: No, 1: Yes
heart_disease = st.radio("Heart Disease", (0, 1))  # 0: No, 1: Yes
ever_married = st.radio("Ever Married", (0, 1))  # 0: No, 1: Yes
work_type = st.selectbox("Work Type", (0, 1, 2, 3, 4))  # 0: Govt_job, 1: Never_worked, 2: Private, 3: Self_employed, 4: children
Residence_type = st.radio("Residence Type", (0, 1))  # 0: Rural, 1: Urban
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", (0, 1, 2, 3))  # 0: unknown, 1: formerly smoked, 2: never smoked, 3: smokes

# Button to make prediction
if st.button('Predict Heart Risk'):
    prediction, user_input = predict_heart_risk(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)
    
    # Display user input in readable form
    st.write("Your input values are:")
    st.write(display_readable_form(user_input))
    
    # Display prediction
    if prediction == 1:
        st.error('High risk of heart-related health issues.')
    elif prediction == 0:
        st.success('Low risk of heart-related health issues.')

# Rest of your Streamlit code goes here
