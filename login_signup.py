import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import xgboost as xgb
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import firestore
import webbrowser
import time
# Initialize Firebase app using your service account key file
cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')

try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app(cred)

db = firestore.client()
def login(key=None):
    # Get user input for email and password
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Log the user in when "Log In" button is clicked
    if st.button("Log In"):
        try:
            user = auth.get_user_by_email(email)
            st.success(f"Welcome back, {user}!")
            return True
        except Exception as e:
            st.error(f"Error logging in: {e}")
            return False

def signup(key=None):
    # Get user input for email, password, and display name
    email = st.text_input("Email", key="signup-email")
    password = st.text_input("Password", type="password", key="signup-password")
    name = st.text_input("Display Name", key="signup-name")

    # Create new user when "Sign Up" button is clicked
    if st.button("Sign Up"):
        try:
            user = auth.create_user(email=email, password=password, display_name=name)
            st.success(f"Welcome, {name}! Your account has been created successfully.")
            return True
        except Exception as e:
            st.error(f"Error creating user: {e}")
            return False
        

def main():
    st.title("AI Fusion Framework")

    # Header with animation
    st.header("Welcome to AI Fusion Framework")
    st.write("Your one-stop solution for predictive analytics.")
    st.balloons()
    login_signup_key = 'login-signup'
    login_key = 'login'
    signup_key = 'signup'

    choice = st.radio("Choose an option", ("Login", "Sign Up"), key=login_signup_key)
    
    if choice == "Login":
        if login(key=login_key):
            # Redirect to the ml_model output in a new tab
            new_tab_url = "http://localhost:8502/"
            webbrowser.open_new_tab(new_tab_url)
            st.experimental_rerun()
            #login(key=login_key)
        
    elif choice == "Sign Up":
        if signup(key=signup_key):
            new_tab_url = "http://localhost:8502/"
            webbrowser.open_new_tab(new_tab_url)
            st.experimental_rerun()



if __name__ == "__main__":
    main()

# def rr():
#     other = util.import_file("ml_model.py")
#     other.run_app()


        ###---- HIDE STREAMLIT STYLE ----


# loading the saved models

# diabetes_model = pickle.load(open('D:/6th Semester/Project/models/diabetes_model.sav', 'rb'))

# heart_disease_model = pickle.load(open('D:/6th Semester/Project/models/heart_disease_model.sav','rb'))

# parkinsons_model = pickle.load(open('D:/6th Semester/Project/models/parkinsons_model.sav', 'rb'))

# medical_insurance_model = pickle.load(open('D:/6th Semester/Project/models/medical_insurance_model.sav', 'rb'))

# cancer_model = pickle.load(open('D:/6th Semester/Project/models/breast_classification_model.sav', 'rb'))

# calories_burnt_model = pickle.load(open('D:/6th Semester/Project/models/calories_burnt_model.sav', 'rb'))

# # sidebar for navigation

# with st.sidebar:
    
#     selected = option_menu('Multiple Disease Prediction System',
                        
#                         ['Diabetes Prediction',
#                         'Heart Disease Prediction',
#                         'Parkinsons Prediction',
#                         'Medical Insuarance Cost Prediction',
#                         'Calories Burnt Prediction',
#                         'Breast Cancer Prediction',
#                         ],
#                         icons=['activity','heart','person','journal-medical','app'],
#                         default_index=0)
    
    
# # Diabetes Prediction Page
# if (selected == 'Diabetes Prediction'):
    
#     # page title
#     st.title('Diabetes Prediction using ML')
    
    
#     # getting the input data from the user
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         Pregnancies = st.text_input('Number of Pregnancies')
        
#     with col2:
#         Glucose = st.text_input('Glucose Level')
    
#     with col3:
#         BloodPressure = st.text_input('Blood Pressure value')
    
#     with col1:
#         SkinThickness = st.text_input('Skin Thickness value')
    
#     with col2:
#         Insulin = st.text_input('Insulin Level')
    
#     with col3:
#         BMI = st.text_input('BMI value')
    
#     with col1:
#         DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
#     with col2:
#         Age = st.text_input('Age of the Person')
    
    
#     # code for Prediction
#     diab_diagnosis = ''
    
#     # creating a button for Prediction
    
#     if st.button('Diabetes Test Result'):
#         diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
#         if (diab_prediction[0] == 1):
#             diab_diagnosis = 'The person is diabetic'
#         else:
#             diab_diagnosis = 'The person is not diabetic'
        
#     st.success(diab_diagnosis)




# # Heart Disease Prediction Page
# if (selected == 'Heart Disease Prediction'):
    
#     # page title
#     st.title('Heart Disease Prediction using ML')
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         age = st.text_input('Age')
        
#     with col2:
#         sex = st.text_input('Sex')
        
#     with col3:
#         cp = st.text_input('Chest Pain types')
        
#     with col1:
#         trestbps = st.text_input('Resting Blood Pressure')
        
#     with col2:
#         chol = st.text_input('Serum Cholestoral in mg/dl')
        
#     with col3:
#         fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
#     with col1:
#         restecg = st.text_input('Resting Electrocardiographic results')
        
#     with col2:
#         thalach = st.text_input('Maximum Heart Rate achieved')
        
#     with col3:
#         exang = st.text_input('Exercise Induced Angina')
        
#     with col1:
#         oldpeak = st.text_input('ST depression induced by exercise')
        
#     with col2:
#         slope = st.text_input('Slope of the peak exercise ST segment')
        
#     with col3:
#         ca = st.text_input('Major vessels colored by flourosopy')
        
#     with col1:
#         thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
    
    
#     # code for Prediction
#     heart_diagnosis = ''
    
#     # creating a button for Prediction
    
#     if st.button('Heart Disease Test Result'):
                            
#         heart_prediction = heart_disease_model.predict([[float(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]])

#         if (heart_prediction[0] == 1):
#             heart_diagnosis = 'The person is having heart disease'
#         else:
#             heart_diagnosis = 'The person does not have any heart disease'
        
#     st.success(heart_diagnosis)
        
    
    

# # Parkinson's Prediction Page
# if (selected == "Parkinsons Prediction"):
    
#     # page title
#     st.title("Parkinson's Disease Prediction using ML")
    
#     col1, col2, col3, col4, col5 = st.columns(5)  
    
#     with col1:
#         fo = st.text_input('MDVP:Fo(Hz)')
        
#     with col2:
#         fhi = st.text_input('MDVP:Fhi(Hz)')
        
#     with col3:
#         flo = st.text_input('MDVP:Flo(Hz)')
        
#     with col4:
#         Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
#     with col5:
#         Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
#     with col1:
#         RAP = st.text_input('MDVP:RAP')
        
#     with col2:
#         PPQ = st.text_input('MDVP:PPQ')
        
#     with col3:
#         DDP = st.text_input('Jitter:DDP')
        
#     with col4:
#         Shimmer = st.text_input('MDVP:Shimmer')
        
#     with col5:
#         Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
#     with col1:
#         APQ3 = st.text_input('Shimmer:APQ3')
        
#     with col2:
#         APQ5 = st.text_input('Shimmer:APQ5')
        
#     with col3:
#         APQ = st.text_input('MDVP:APQ')
        
#     with col4:
#         DDA = st.text_input('Shimmer:DDA')
        
#     with col5:
#         NHR = st.text_input('NHR')
        
#     with col1:
#         HNR = st.text_input('HNR')
        
#     with col2:
#         RPDE = st.text_input('RPDE')
        
#     with col3:
#         DFA = st.text_input('DFA')
        
#     with col4:
#         spread1 = st.text_input('spread1')
        
#     with col5:
#         spread2 = st.text_input('spread2')
        
#     with col1:
#         D2 = st.text_input('D2')
        
#     with col2:
#         PPE = st.text_input('PPE')
        
    
    
#     # code for Prediction
#     parkinsons_diagnosis = ''
    
#     # creating a button for Prediction    
#     if st.button("Parkinson's Test Result"):
#         parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
#         if (parkinsons_prediction[0] == 1):
#             parkinsons_diagnosis = "The person has Parkinson's disease"
#         else:
#             parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
#     st.success(parkinsons_diagnosis)

# # Medical Insuarance Prediction Page
# if (selected == 'Medical Insuarance Cost Prediction'):
    
#     # page title
#     st.title('Medical Insuarance Cost Prediction using ML')
    
    
#     # getting the input data from the user
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.text_input('age')

#     with col2:
#         sex = st.text_input('sex')

#     with col3:
#         bmi = st.text_input('bmi')

#     with col1:
#         children = st.text_input('children')

#     with col2:
#         smoker = st.text_input('smoker')

#     with col3:
#         region = st.text_input('region')

#     # code for Prediction
#     medical_insurance_model_diagnosis = ''

#     # creating a button for Prediction
#     if st.button('Medical Insurance Cost Result'):
#         if age and bmi and children and sex and smoker and region:
#             age = int(age)  # convert age to an integer
#             bmi = float(bmi)  # convert bmi to a float
#             children = int(children)  # convert children to an integer
#             # convert sex to an integer (0 for female, 1 for male)
#             sex = 0 if sex.lower() == 'male' else 1  
#             # convert smoker to an integer (0 for non-smoker, 1 for smoker)
#             smoker = 0 if smoker.lower() == 'yes' else 1  
#             # convert region to an integer (0 for northeast, 1 for northwest, 2 for southeast, 3 for southwest)
#             region = {'northeast': 2, 'northwest': 3, 'southeast': 0, 'southwest': 1}[region.lower()]
#             medical_insurance_model_prediction = medical_insurance_model.predict([[age, sex, bmi, children, smoker, region]])
#             medical_insurance_model_diagnosis = 'The insurance cost is USD ' + str(medical_insurance_model_prediction[0])
            
#     st.success(medical_insurance_model_diagnosis)


#     # Calories Burnt Prediction Page
# if (selected == 'Calories Burnt Prediction'):

#     # page title
#     st.title('Calories Burnt Prediction using ML')

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         user_id = st.text_input('User ID')

#     with col2:
#         gender = st.selectbox('Gender', ['Male', 'Female'])

#     with col3:
#         age = st.number_input('Age', min_value=1, max_value=150)

#     with col1:
#         height = st.number_input('Height (inches)', min_value=1, max_value=120)

#     with col2:
#         weight = st.number_input('Weight (lbs)', min_value=1, max_value=1000)

#     with col3:
#         duration = st.number_input('Duration (minutes)', min_value=1, max_value=1440)

#     with col1:
#         heart_rate = st.number_input('Heart Rate (bpm)', min_value=1, max_value=250)

#     with col2:
#         body_temp = st.number_input('Body Temperature (F)', min_value=70, max_value=110)

#     with col3:
#         calories = st.number_input('Calories', min_value=1, max_value=10000)

#     # code for Prediction
#     calories_diagnosis = ''

#     # creating a button for Prediction

#     if st.button('Calories Burnt Test Result'):
#         # creating a list of input parameters
#         input_data = [[user_id, gender, age, height, weight, duration, heart_rate, body_temp, calories]]
#         input_str = ' '.join(map(str, input_data))
#         input_str = input_str.encode('utf-8')
#         input_data = [str(i) for i in input_data]  # Convert input data to strings
#         input_data = [s.encode('ascii', 'ignore') for s in input_data]  # Convert to ASCII format

#         # calling the pre-trained ML model for prediction
#         calories_prediction = calories_burnt_model.predict(input_data)

#         if (calories_prediction[0] == 1):
#             calories_diagnosis = 'The calories burnt value is predicted to be higher than average.'
#         else:
#             calories_diagnosis = 'The calories burnt value is predicted to be lower than average.'

#     st.success(calories_diagnosis)

# # Breast Cancer Prediction Page
# if (selected == 'Breast Cancer Prediction'):

#     # page title
#     st.title('Breast Cancer Prediction using ML')

#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         radius_mean = st.text_input('Mean Radius')

#         texture_mean = st.text_input('Mean Texture')

#         perimeter_mean = st.text_input('Mean Perimeter')

#         area_mean = st.text_input('Mean Area')

#         smoothness_mean = st.text_input('Mean Smoothness')

#         compactness_mean = st.text_input('Mean Compactness')

#         concavity_mean = st.text_input('Mean Concavity')

#         concave_points_mean = st.text_input('Mean Concave Points')

    

#     with col2:
#         symmetry_mean = st.text_input('Mean Symmetry')

#         fractal_dimension_mean = st.text_input('Mean Fractal Dimension')
        
#         radius_se = st.text_input('Standard Error Radius')

#         texture_se = st.text_input('Standard Error Texture')

#         perimeter_se = st.text_input('Standard Error Perimeter')

#         area_se = st.text_input('Standard Error Area')

#         smoothness_se = st.text_input('Standard Error Smoothness')

#         compactness_se = st.text_input('Standard Error Compactness')

        

        

#     with col3:
#         concavity_se = st.text_input('Standard Error Concavity')

#         concave_points_se = st.text_input('Standard Error Concave Points')
        
#         symmetry_se = st.text_input('Standard Error Symmetry')

#         fractal_dimension_se = st.text_input('Standard Error Fractal Dimension')
        
#         radius_worst = st.text_input('Worst Radius')

#         texture_worst = st.text_input('Worst Texture')

#         perimeter_worst = st.text_input('Worst Perimeter')

#         area_worst = st.text_input('Worst Area')

        

#     with col4:
#         smoothness_worst = st.text_input('Worst Smoothness')

#         compactness_worst = st.text_input('Worst Compactness')

#         concavity_worst = st.text_input('Worst Concavity')

#         concave_points_worst = st.text_input('Worst Concave Points')

#         symmetry_worst = st.text_input('Worst Symmetry')

#         fractal_dimension_worst = st.text_input('Worst Fractal Dimension')


#     # code for Prediction
#     cancer_diagnosis = ''

#     # creating a button for Prediction

#     if st.button('Breast Cancer Test Result'):

#         cancer_prediction = cancer_model.predict([[float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean), float(smoothness_mean), float(compactness_mean), float(concavity_mean), float(concave_points_mean), float(symmetry_mean), float(fractal_dimension_mean),
#                                                     float(radius_se), float(texture_se), float(perimeter_se), float(area_se), float(smoothness_se), float(compactness_se), float(concavity_se), float(concave_points_se), float(symmetry_se), float(fractal_dimension_se),
#                                                     float(radius_worst), float(texture_worst), float(perimeter_worst), float(area_worst), float(smoothness_worst), float(compactness_worst), float(concavity_worst), float(concave_points_worst), float(symmetry_worst), float(fractal_dimension_worst)]])

#         if (cancer_prediction[0] == 1):
#             cancer_diagnosis = 'The tumor is benign'
#         else:
#             cancer_diagnosis = 'The tumor is malignant'

#     st.success(cancer_diagnosis)
