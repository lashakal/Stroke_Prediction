import streamlit as st
import prediction_model

st.title('Stroke Prediction  :heart:')
st.caption('Created by:   Amadin Ahmed   Lasha Kaliashvili   Mikheil Uglava')
st.text("")
st.markdown("According to the World Health Organization (WHO), stroke is the leading cause of disability worldwide and the second leading cause of death. Over the last 17years, the lifetime risk of developing a stroke has increased by 50% and now 1 in 4 people is estimated to have a stroke in their lifetime. In this project, we are aiming to design a stroke prediction model with a user-friendly interface that can assess an individual’s risk of stroke based on their health data and medical history using the KNN classification algorithm. ")
st.text("")



def predict_stroke_risk(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    # Map the string inputs to their corresponding numerical values
    gender_num = 0 if gender == "Female" else 1
    hypertension_num = 0 if hypertension == "No" else 1
    heart_disease_num = 0 if heart_disease == "No" else 1
    ever_married_num = 0 if ever_married == "No" else 1

    work_type_map = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self_employed": 3, "Takes Care of Childern": 4}
    work_type_num = work_type_map[work_type]

    Residence_type_num = 0 if Residence_type == "Rural" else 1

    smoking_status_map = {"unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}
    smoking_status_num = smoking_status_map[smoking_status]

    # Create a 2D array (or list) for the input data
    user_data = [gender_num, age, hypertension_num, heart_disease_num, ever_married_num, work_type_num, Residence_type_num, avg_glucose_level, bmi, smoking_status_num]

    # Use the model to make a prediction
    prediction = prediction_model.KNN(user_data)
    return prediction, user_data

    # return user_data[0], user_data



# User interface in Streamlit to collect user inputs
gender = st.radio("Gender", ("Female", "Male"))  # 0: Female, 1: Male
age = st.number_input("Age", min_value=0)
hypertension = st.radio("Hypertension", ("No", "Yes"))  # 0: No, 1: Yes
heart_disease = st.radio("Heart Disease", ("No", "Yes"))  # 0: No, 1: Yes
ever_married = st.radio("Ever Married", ("No", "Yes"))  # 0: No, 1: Yes
work_type = st.selectbox("Work Type", ("Govt_job", "Never_worked", "Private", "Self_employed", "Takes Care of Childern"))  # 0: Govt_job, 1: Never_worked, 2: Private, 3: Self_employed, 4: children
Residence_type = st.radio("Residence Type", ("Rural", "Urban"))  # 0: Rural, 1: Urban
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ("unknown", "formerly smoked", "never smoked", "smokes"))  # 0: unknown, 1: formerly smoked, 2: never smoked, 3: smokes


print("Changed User_Input: ",gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)
 
# Button to make prediction
if st.button('Predict Stroke Risk'):
    prediction, user_input = predict_stroke_risk(gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)
    
    # Display user input in readable form
    st.success(user_input)
    
    
    # Display prediction
    if prediction == 1:
       st.error('High risk of stroke-related health issues.')
    elif prediction == 0:
       st.success('Low risk of stroke-related health issues.')
