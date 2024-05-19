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
from firebase_admin import credentials, auth, db
from firebase_admin import firestore
from st_pages import Page, show_pages, add_page_title


# initialize the Firebase app for Heart Disease Prediction model
def initialize_heart_disease_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    
    try:
        app = firebase_admin.get_app('heart_disease_app')
    except ValueError:
        app = firebase_admin.initialize_app(cred, name='heart_disease_app')
    heart_db = firestore.client(app)
    return heart_db

heart_db = initialize_heart_disease_app()

# initialize the Firebase app for Diabetes Prediction model
def initialize_diabetes_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('diabetes_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='diabetes_app')
    diabetes_db = firestore.client(app)
    return diabetes_db 

diabetes_db = initialize_diabetes_app()

# initialize the Firebase app for Parkinsons Prediction model
def initialize_parkinsons_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('parkinsons_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='parkinsons_app')
    parkinsons_db = firestore.client(app)
    return parkinsons_db

parkinsons_db = initialize_parkinsons_app()

# initialize the Firebase app for Medical Insuarance Prediction model
def initialize_medical_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('medical_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='medical_app')
    medical_db = firestore.client(app)
    return medical_db

medical_db = initialize_medical_app()

# initialize the Firebase app for Breast Cancer Prediction model
def initialize_breast_cancer_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('breast_cancer_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='breast_cancer_app')
    breast_cancer_db = firestore.client(app)
    return breast_cancer_db

breast_cancer_db = initialize_breast_cancer_app()

# initialize the Firebase app for house price Prediction model
def initialize_house_price_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('house_price_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='house_price_app')
    house_price_db = firestore.client(app)
    return house_price_db

house_price_db = initialize_house_price_app()

# initialize the Firebase app for gold price Prediction model
def initialize_gold_price_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('gold_price_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='gold_price_app')
    gold_price_db = firestore.client(app)
    return gold_price_db

gold_price_db = initialize_house_price_app()

# initialize the Firebase app for loan Prediction model
def initialize_loan_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('loan_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='loan_app')
    loan_db = firestore.client(app)
    return loan_db

loan_db = initialize_house_price_app()

# initialize the Firebase app for car Prediction model
def initialize_car_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\SEMINAR\\models\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('car_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='car_app')
    car_db = firestore.client(app)
    return car_db

car_db = initialize_house_price_app()

# def run_ml_app():
#     st.title("qwerty")
#     st.empty()
    
# if __name__ == "__main__":
#     run_ml_app()

# def main():
#     run_ml_app()
    
def run_ml_app():
    
    st.set_page_config(
        page_title="WealthCare Predictive Suite",
        page_icon="	:robot_face:",
        layout="wide",
    )
    diabetes_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/diabetes_model.sav', 'rb'))

    heart_disease_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/heart_disease_model.sav','rb'))

    parkinsons_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/parkinsons_model.sav', 'rb'))

    medical_insurance_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/medical_insurance_model.sav', 'rb'))

    cancer_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/breast_classification_model.sav', 'rb'))
    
    house_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/house_price_prediction_model.sav', 'rb'))
    
    gold_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/gold_price_prediction_model.sav', 'rb'))
    
    loan_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/loan_prediction_model.sav', 'rb'))
    
    car_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/car_prediction_model.sav', 'rb'))

    calories_burnt_model = pickle.load(open('D:/#YPR/3rd Trimester/SEMINAR/models/calories_burnt_model.sav', 'rb'))

    # sidebar for navigation

    with st.sidebar:
        
        selected = option_menu('WealthCare Predictive Suite',
                            
                            ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Medical Insuarance Cost Prediction',
                            'Breast Cancer Prediction',
                            'House Price Prediction',
                            'Gold Price Prediction',
                            'Loan Price Prediction',
                            'Car Price Prediction'
                            ],
                            icons=['activity','heart','person','journal-medical','hospital','house','cash','bank','ev-front'],
                            default_index=0)
         # Navigation and main content
    if selected == 'Home':
        st.title('Welcome to the ML Prediction System')
        st.write('This system provides predictions for various diseases and financial metrics.')
        st.write('Please select an option from the sidebar to get started.')
    
        
    # House Price Prediction Page
    if selected == 'House Price Prediction':
        # Page title
        st.title('House Price Prediction using ML')

        # Input fields
        col1, col2, col3 = st.columns(3)
        with col1:
            CRIM = st.text_input('CRIM: per capita crime rate by town')
            ZN = st.text_input('ZN: proportion of residential land zoned for lots over 25,000 sq.ft.')
            INDUS = st.text_input('INDUS: proportion of non-retail business acres per town')
            CHAS = st.text_input('CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)')
            NOX = st.text_input('NOX: nitric oxides concentration (parts per 10 million) [parts/10M]')
            RM = st.text_input('RM: average number of rooms per dwelling')
        with col2:
            AGE = st.text_input('AGE: proportion of owner-occupied units built prior to 1940')
            DIS = st.text_input('DIS: weighted distances to five Boston employment centres')
            RAD = st.text_input('RAD: index of accessibility to radial highways')
            TAX = st.text_input('TAX: full-value property-tax rate per $10,000 [$/10k]')
            PTRATIO = st.text_input('PTRATIO: pupil-teacher ratio by town')
            B = st.text_input('B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town')
        with col3:
            LSTAT = st.text_input('LSTAT: % lower status of the population')

        # Prediction logic
        price_prediction = ''
        if st.button('Predict House Price'):
            # Convert input to float
            CRIM = float(CRIM)
            ZN = float(ZN)
            INDUS = float(INDUS)
            CHAS = float(CHAS)
            NOX = float(NOX)
            RM = float(RM)
            AGE = float(AGE)
            DIS = float(DIS)
            RAD = float(RAD)
            TAX = float(TAX)
            PTRATIO = float(PTRATIO)
            B = float(B)
            LSTAT = float(LSTAT)
            
            # Predict house price
            price_prediction = house_model.predict([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

            # Store input data in the database
            input_data = {
                'CRIM': CRIM,
                'ZN': ZN,
                'INDUS': INDUS,
                'CHAS': CHAS,
                'NOX': NOX,
                'RM': RM,
                'AGE': AGE,
                'DIS': DIS,
                'RAD': RAD,
                'TAX': TAX,
                'PTRATIO': PTRATIO,
                'B': B,
                'LSTAT': LSTAT,
                'predicted_price': float(price_prediction[0])
            }
            house_price_db.collection('house_price_inputs').add(input_data)
            st.success(f'Predicted house price: {price_prediction}''Thousand Dollars')
            
            
                # House Price Prediction Page
    if selected == 'Car Price Prediction':
        # Page title
        st.title('Car Price Prediction using ML')

        # Input fields
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            Year = st.number_input('Year')
        with col2:
            Present_Price = st.number_input('Present Price')
        with col3:
            Kms_Driven = st.number_input('Kms Driven')
        with col4:
            Fuel_Type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
        with col5:
            Seller_Type = st.selectbox('Seller Type', ['Dealer', 'Individual'])

        col6, col7, col8 = st.columns(3)
        with col6:
            Transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
        with col7:
            Owner = st.selectbox('Owner', [0, 1, 2, 3])
        with col8:
            Car_Name = st.text_input('Car Name')


        # convert categorical columns to numerical values
        fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
        seller_map = {'Dealer': 0, 'Individual': 1}
        transmission_map = {'Manual': 0, 'Automatic': 1}

        Fuel_Type = fuel_map[Fuel_Type]
        Seller_Type = seller_map[Seller_Type]
        Transmission = transmission_map[Transmission]

        # Prediction logic
        car_price_prediction = ''
        if st.button('Predict Car Price'):
            # Predict car price
            car_price_prediction = car_model.predict([[Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner]])

            # Store input data in the database
            input_data = {
                'Year': int(Year),
                'Present_Price': float(Present_Price),
                'Kms_Driven': int(Kms_Driven),
                'Fuel_Type': int(Fuel_Type),
                'Seller_Type': int(Seller_Type),
                'Transmission': int(Transmission),
                'Owner': int(Owner),
                'Car_Name': Car_Name,
                'predicted_car_price': float(car_price_prediction[0])
            }
            car_db.collection('car_price_inputs').add(input_data)
            st.success(f'Predicted car price: {car_price_prediction}')

        
            
    # Loan Price Prediction Page
    if selected == 'Loan Price Prediction':
        # Page title
        st.title('Loan Price Prediction using SVM')

        # Input fields
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            Gender = st.selectbox('Gender', ['Male', 'Female'])
        with col2:
            Married = st.selectbox('Married', ['No', 'Yes'])
        with col3:
            Dependents = st.selectbox('Dependents', ['0', '1', '2', '3', '4'])
        with col4:
            Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
        with col5:
            Self_Employed = st.selectbox('Self Employed', ['No', 'Yes'])

        col6, col7, col8, col9 = st.columns(4)
        with col6:
            ApplicantIncome = st.number_input('Applicant Income')
        with col7:
            CoapplicantIncome = st.number_input('Coapplicant Income')
        with col8:
            LoanAmount = st.number_input('Loan Amount')
        with col9:
            Loan_Amount_Term = st.number_input('Loan Amount Term')

        col10, col11 = st.columns(2)
        with col10:
            Credit_History = st.selectbox('Credit History', [0, 1])
        with col11:
            Property_Area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])


                # convert categorical columns to numerical values
        gender_map = {'Male': 1, 'Female': 0}
        married_map = {'No': 0, 'Yes': 1}
        dependents_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        education_map = {'Graduate': 1, 'Not Graduate': 0}
        self_employed_map = {'No': 0, 'Yes': 1}
        property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

        Gender = gender_map[Gender]
        Married = married_map[Married]
        Dependents = dependents_map[Dependents]
        Education = education_map[Education]
        Self_Employed = self_employed_map[Self_Employed]
        Property_Area = property_area_map[Property_Area]

        # Prediction logic
        loan_price_prediction = ''
        if st.button('Predict Loan Price'):
            # Predict loan price
            loan_price_prediction = loan_model.predict([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])

            # Store input data in the database
            input_data = {
                'Gender': int(Gender),
                'Married': int(Married),
                'Dependents': int(Dependents),
                'Education': int(Education),
                'Self_Employed': int(Self_Employed),
                'ApplicantIncome': int(ApplicantIncome),
                'CoapplicantIncome': int(CoapplicantIncome),
                'LoanAmount': int(LoanAmount),
                'Loan_Amount_Term': int(Loan_Amount_Term),
                'Credit_History': int(Credit_History),
                'Property_Area': int(Property_Area),
                'predicted_loan_price': int(loan_price_prediction[0])
            }
            loan_db.collection('loan_price_inputs').add(input_data)
            st.success(f'Predicted loan price: {loan_price_prediction}')

            
    # Gold Price Prediction Page
    if selected == 'Gold Price Prediction':
        # Page title
        st.title('Gold Price Prediction using ML')

        # Input fields
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            SPX = st.text_input('SPX: S&P 500')
        with col2:
            USO = st.text_input('USO: US Oil Fund')
        with col3:
            SLV = st.text_input('SLV: Silver Price')
        with col4:
            EUR_USD = st.text_input('EUR/USD: Euro to US Dollar Exchange Rate')

        # Prediction logic
        gold_price_prediction = ''
        if st.button('Predict Gold Price'):
            # Convert input to appropriate data type
            SPX = float(SPX)
            USO = float(USO)
            SLV = float(SLV)
            EUR_USD = float(EUR_USD)

            # Predict gold price
            gold_price_prediction = gold_model.predict([[SPX, USO, SLV, EUR_USD]])

            # Store input data in the database
            input_data = {
                
                'SPX': SPX,
                'USO': USO,
                'SLV': SLV,
                'EUR/USD': EUR_USD,
                'predicted_price': float(gold_price_prediction[0])  # Convert prediction to float
            }
            gold_price_db.collection('gold_price_inputs').add(input_data)
            st.success(f'Predicted gold price: {gold_price_prediction}')

                    
          
    # Diabetes Prediction Page
    if (selected == 'Diabetes Prediction'):
        
        # page title
        st.title('Diabetes Prediction using ML')
        
        
        # getting the input data from the user
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
            
            
        with col2:
            Glucose = st.text_input('Glucose Level(80-200mg/dL)')
        
        with col3:
            BloodPressure = st.text_input('Blood Pressure value()')
        
        with col1:
            SkinThickness = st.text_input('Skin Thickness value')
        
        with col2:
            Insulin = st.text_input('Insulin Level')
        
        with col3:
            BMI = st.text_input('BMI value')
        
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
        with col2:
            Age = st.text_input('Age of the Person')
        
        
        # code for Prediction
        diab_diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Diabetes Test Result'):
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            # create a dictionary to store the input data
            input_data = {
                'pregnancies': Pregnancies,
                'glucose': Glucose,
                'blood_pressure': BloodPressure,
                'skin_thickness': SkinThickness,
                'insulin': Insulin,
                'bmi': BMI,
                'dpf': DiabetesPedigreeFunction,
                'age': Age,
                'diagnosis': diab_prediction[0]
            }
            input_data = {k: int(v) if isinstance(v, np.int64) else v for k, v in input_data.items()}
            # 'pregnancies': This attribute represents the number of times a patient has been pregnant. It is a known risk factor for developing diabetes in women.

            # 'glucose': This attribute represents the patient's fasting plasma glucose levels. High levels of glucose in the blood are a common symptom of diabetes.

            # 'blood_pressure': This attribute represents the patient's diastolic blood pressure. High blood pressure is a risk factor for developing diabetes.

            # 'skin_thickness': This attribute represents the thickness of the patient's skinfold at the triceps. Higher values may be associated with insulin resistance and diabetes.

            # 'insulin': This attribute represents the patient's insulin levels. Insulin is responsible for regulating glucose levels in the blood, and low levels may indicate diabetes.

            # 'bmi': This attribute represents the patient's body mass index, which is a measure of body fat based on height and weight. Obesity is a major risk factor for developing diabetes.

            # 'dpf': This attribute represents the patient's diabetes pedigree function, which is a measure of the patient's genetic risk for developing diabetes.

            # 'age': This attribute represents the patient's age. As people age, their risk of developing diabetes increases.

            # 'diagnosis': This attribute represents the predicted diagnosis of the patient (either "diabetic" or "non-diabetic") based on the other attributes in the dataset and the trained model.
            
            # store the input data in the database
            diabetes_db.collection('diabetes_inputs').add(input_data)
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
            
        st.success(diab_diagnosis)




    # Heart Disease Prediction Page
    if (selected == 'Heart Disease Prediction'):
        
        # initialize the Firebase app and Firestore database for Heart Disease Prediction model
        db = initialize_heart_disease_app()
        # page title
        st.title('Heart Disease Prediction using ML')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.text_input('Age')
            
        with col2:
            sex = st.text_input('Sex')
            
        with col3:
            cp = st.text_input('Chest Pain types')
            
        with col1:
            trestbps = st.text_input('Resting Blood Pressure')
            
        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')
            
        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
            
        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')
            
        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')
            
        with col3:
            exang = st.text_input('Exercise Induced Angina')
            
        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')
            
        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')
            
        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')
            
        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
            
        
        # code for Prediction
        heart_diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Heart Disease Test Result'):
                                
            heart_prediction = heart_disease_model.predict([[float(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]])

            if (heart_prediction[0] == 1):
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
                
             # create a dictionary to store the input data
            input_data = {
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal,
                'diagnosis': heart_diagnosis
            }
            input_data = {k: int(v) if isinstance(v, np.int64) else v for k, v in input_data.items()}
            # 'age': This attribute represents the age of the patient. Age is a critical factor in heart disease, and as people get older, their risk of heart disease increases.

            # 'sex': This attribute represents the gender of the patient. Men are at a higher risk of heart disease than women, especially at younger ages.

            # 'cp': This attribute represents the type of chest pain experienced by the patient. Different types of chest pain can be indicative of different types of heart problems.

            # 'trestbps': This attribute represents the resting blood pressure of the patient. High blood pressure is a significant risk factor for heart disease.

            # 'chol': This attribute represents the cholesterol levels of the patient. High cholesterol levels can lead to a buildup of plaque in the arteries, increasing the risk of heart disease.

            # 'fbs': This attribute represents the fasting blood sugar levels of the patient. High levels of blood sugar can lead to diabetes, which increases the risk of heart disease.

            # 'restecg': This attribute represents the results of a resting electrocardiogram, which measures the electrical activity of the heart. Abnormal results can indicate underlying heart problems.

            # 'thalach': This attribute represents the maximum heart rate achieved during exercise. A lower maximum heart rate can indicate poor cardiovascular fitness, which is associated with a higher risk of heart disease.

            # 'exang': This attribute represents whether the patient experiences exercise-induced angina, which can be a symptom of underlying heart problems.

            # 'oldpeak': This attribute represents the ST depression induced by exercise relative to rest. ST depression can be a sign of coronary artery disease.

            # 'slope': This attribute represents the slope of the peak exercise ST segment. Different slopes can indicate different types of heart problems.

            # 'ca': This attribute represents the number of major vessels colored by fluoroscopy. A higher number of colored vessels can indicate a higher risk of heart disease.

            # 'thal': This attribute represents the type of thalassemia the patient has, which is a genetic disorder that affects the production of hemoglobin. Thalassemia can increase the risk of heart problems.
            
            
            # store the input data in the database
            heart_db.collection('Heart_inputs').add(input_data)
            
        st.success(heart_diagnosis)
            
        
        

    # Parkinson's Prediction Page
    if (selected == "Parkinsons Prediction"):
        
        # page title
        st.title("Parkinson's Disease Prediction using ML")
        
        col1, col2, col3, col4, col5 = st.columns(5)  
        
        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')
            
        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)')
            
        with col3:
            flo = st.text_input('MDVP:Flo(Hz)')
            
        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)')
            
        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
            
        with col1:
            RAP = st.text_input('MDVP:RAP')
            
        with col2:
            PPQ = st.text_input('MDVP:PPQ')
            
        with col3:
            DDP = st.text_input('Jitter:DDP')
            
        with col4:
            Shimmer = st.text_input('MDVP:Shimmer')
            
        with col5:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
            
        with col1:
            APQ3 = st.text_input('Shimmer:APQ3')
            
        with col2:
            APQ5 = st.text_input('Shimmer:APQ5')
            
        with col3:
            APQ = st.text_input('MDVP:APQ')
            
        with col4:
            DDA = st.text_input('Shimmer:DDA')
            
        with col5:
            NHR = st.text_input('NHR')
            
        with col1:
            HNR = st.text_input('HNR')
            
        with col2:
            RPDE = st.text_input('RPDE')
            
        with col3:
            DFA = st.text_input('DFA')
            
        with col4:
            spread1 = st.text_input('spread1')
            
        with col5:
            spread2 = st.text_input('spread2')
            
        with col1:
            D2 = st.text_input('D2')
            
        with col2:
            PPE = st.text_input('PPE')
            
        
        
        # code for Prediction
        parkinsons_diagnosis = ''
        
        # creating a button for Prediction    
        if st.button("Parkinson's Test Result"):
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
            
            if (parkinsons_prediction[0] == 1):
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
            input_data= {
                'MDVP:Fo(Hz)': fo,
                'MDVP:Fhi(Hz)': fhi,
                'MDVP:Flo(Hz)': flo,
                'MDVP:Jitter(%)': Jitter_percent,
                'MDVP:Jitter(Abs)': Jitter_Abs,
                'MDVP:RAP': RAP,
                'MDVP:PPQ': PPQ,
                'Jitter:DDP': DDP,
                'MDVP:Shimmer': Shimmer,
                'MDVP:Shimmer(dB)': Shimmer_dB,
                'Shimmer:APQ3': APQ3,
                'Shimmer:APQ5': APQ5,
                'MDVP:APQ': APQ,
                'Shimmer:DDA': DDA,
                'NHR': NHR,
                'HNR': HNR,
                'RPDE': RPDE,
                'DFA': DFA,
                'spread1': spread1,
                'spread2': spread2,
                'D2': D2,
                'PPE': PPE
            }
            input_data = {k: int(v) if isinstance(v, np.int64) else v for k, v in input_data.items()}
            # 'MDVP:Fo(Hz)': fundamental frequency, the base rate of vibration of the vocal folds
            # 'MDVP:Fhi(Hz)': highest frequency in the spectrum
            # 'MDVP:Flo(Hz)': lowest frequency in the spectrum
            # 'MDVP:Jitter(%)': variation in frequency, measured as a percentage
            # 'MDVP:Jitter(Abs)': absolute value of the variation in frequency
            # 'MDVP:RAP': rate of change of frequency
            # 'MDVP:PPQ': variation in frequency using a different calculation method
            # 'Jitter:DDP': average difference between 3 consecutive periods
            # 'MDVP:Shimmer': variation in amplitude or intensity
            # 'MDVP:Shimmer(dB)': variation in amplitude measured in decibels
            # 'Shimmer:APQ3': amplitude perturbation quotient
            # 'Shimmer:APQ5': amplitude perturbation quotient using 5-point method
            # 'MDVP:APQ': variation in amplitude using a different calculation method
            # 'Shimmer:DDA': average difference between 3 consecutive periods in the amplitude
            # 'NHR': ratio of noise to tonal components in the voice
            # 'HNR': ratio of the energy in the voice signal to the energy in the noise signal
            # 'RPDE': recurrence period density entropy, a measure of the unpredictability of the time between the recurrence of certain patterns in the voice
            # 'DFA': detrended fluctuation analysis, a measure of the fractal scaling properties of the voice signal
            # 'spread1': measure of the variation of the nonlinear features of the voice
            # 'spread2': another measure of the variation of the nonlinear features of the voice
            # 'D2': correlation dimension, a measure of the complexity of the voice signal
            # 'PPE': perturbance entropy, a measure of the amount of disorder in the voice signal.
            # store the input data in the database
            parkinsons_db.collection('parkinsons_inputs').add(input_data)
            
        st.success(parkinsons_diagnosis)

    # Medical Insuarance Prediction Page
    if (selected == 'Medical Insuarance Cost Prediction'):
        
        # page title
        st.title('Medical Insuarance Cost Prediction using ML')
        
        
        # getting the input data from the user
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('age')

        with col2:
            sex = st.text_input('sex')

        with col3:
            bmi = st.text_input('bmi')

        with col1:
            children = st.text_input('children')

        with col2:
            smoker = st.text_input('smoker')

        with col3:
            region = st.text_input('region')

        # code for Prediction
        medical_insurance_model_diagnosis = ''

        # creating a button for Prediction
        if st.button('Medical Insurance Cost Result'):
            if age and bmi and children and sex and smoker and region:
                age = int(age)  # convert age to an integer
                bmi = float(bmi)  # convert bmi to a float
                children = int(children)  # convert children to an integer
                # convert sex to an integer (0 for female, 1 for male)
                sex = 0 if sex.lower() == 'male' else 1  
                # convert smoker to an integer (0 for non-smoker, 1 for smoker)
                smoker = 0 if smoker.lower() == 'yes' else 1  
                # convert region to an integer (0 for northeast, 1 for northwest, 2 for southeast, 3 for southwest)
                region = {'northeast': 2, 'northwest': 3, 'southeast': 0, 'southwest': 1}[region.lower()]
                medical_insurance_model_prediction = medical_insurance_model.predict([[age, sex, bmi, children, smoker, region]])
                medical_insurance_model_diagnosis = 'The insurance cost is USD ' + str(medical_insurance_model_prediction[0])
                
            # create data input store dictionary
            input_data = {
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'children': children,
                'smoker': smoker,
                'region': region
            }
            # 'age': Refers to the age of the person, which can be a significant factor in determining health risks and insurance premiums.
            # 'sex': Refers to the gender of the person, which can also be a factor in determining health risks and insurance premiums.
            # 'bmi': Refers to the body mass index of the person, which is a measure of body fat based on height and weight. It can be a significant factor in determining health risks and insurance premiums.
            # 'children': Refers to the number of children the person has, which can affect their insurance needs and costs.
            # 'smoker': Refers to whether the person is a smoker or not, which can have a significant impact on their health and insurance premiums.
            # 'region': Refers to the geographical region where the person resides, which can affect the availability and cost of healthcare services and insurance options.
                        
            
            # store the input data in the database
            medical_db.collection('medical_insurance_inputs').add(input_data)
        st.success(medical_insurance_model_diagnosis)


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
    #         height = st.number_input('Height (inches)', min_value=1, max_value=250)

    #     with col2:
    #         weight = st.number_input('Weight (lbs)', min_value=1, max_value=1000)

    #     with col3:
    #         duration = st.number_input('Duration (minutes)', min_value=1, max_value=1440)

    #     with col1:
    #         heart_rate = st.number_input('Heart Rate (bpm)', min_value=1, max_value=250)

    #     with col2:
    #         body_temp = st.number_input('Body Temperature (F)', min_value=30, max_value=110)


    #     # code for Prediction
    #     calories_diagnosis = ''

    #     # creating a button for Prediction

    #     if st.button('Calories Burnt Test Result'):
    #         # creating a list of input parameters
    #         input_data = [[gender, age, height, weight, duration, heart_rate, body_temp]]
    #         # Convert the list to a numpy array
    #         input_data = np.array(input_data)
    #         # Reshape the numpy array
    #         input_data = input_data.reshape(1, -1)
    #         input_data = input_data.astype(str) # convert to string
    #         input_data = np.char.encode(input_data, 'ascii') # encode using ASCII encoding
    #         # calling the pre-trained ML model for prediction
    #         calories_prediction = calories_burnt_model.predict(input_data)

    #         if (calories_prediction[0] == 1):
    #             calories_diagnosis = 'The calories burnt value is predicted to be higher than average.'
    #         else:
    #             calories_diagnosis = 'The calories burnt value is predicted to be lower than average.'

    #     st.success(calories_diagnosis)



    # Breast Cancer Prediction Page
    if (selected == 'Breast Cancer Prediction'):

        # page title
        st.title('Breast Cancer Prediction using ML')

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            radius_mean = st.text_input('Mean Radius')

            texture_mean = st.text_input('Mean Texture')

            perimeter_mean = st.text_input('Mean Perimeter')

            area_mean = st.text_input('Mean Area')

            smoothness_mean = st.text_input('Mean Smoothness')

            compactness_mean = st.text_input('Mean Compactness')

            concavity_mean = st.text_input('Mean Concavity')

            concave_points_mean = st.text_input('Mean Concave Points')

        

        with col2:
            symmetry_mean = st.text_input('Mean Symmetry')

            fractal_dimension_mean = st.text_input('Mean Fractal Dimension')
            
            radius_se = st.text_input('Standard Error Radius')

            texture_se = st.text_input('Standard Error Texture')

            perimeter_se = st.text_input('Standard Error Perimeter')

            area_se = st.text_input('Standard Error Area')

            smoothness_se = st.text_input('Standard Error Smoothness')

            compactness_se = st.text_input('Standard Error Compactness')

            

            

        with col3:
            concavity_se = st.text_input('Standard Error Concavity')

            concave_points_se = st.text_input('Standard Error Concave Points')
            
            symmetry_se = st.text_input('Standard Error Symmetry')

            fractal_dimension_se = st.text_input('Standard Error Fractal Dimension')
            
            radius_worst = st.text_input('Worst Radius')

            texture_worst = st.text_input('Worst Texture')

            perimeter_worst = st.text_input('Worst Perimeter')

            area_worst = st.text_input('Worst Area')

            

        with col4:
            smoothness_worst = st.text_input('Worst Smoothness')

            compactness_worst = st.text_input('Worst Compactness')

            concavity_worst = st.text_input('Worst Concavity')

            concave_points_worst = st.text_input('Worst Concave Points')

            symmetry_worst = st.text_input('Worst Symmetry')

            fractal_dimension_worst = st.text_input('Worst Fractal Dimension')


        # code for Prediction
        cancer_diagnosis = ''

        # creating a button for Prediction

        if st.button('Breast Cancer Test Result'):

            cancer_prediction = cancer_model.predict([[float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean), float(smoothness_mean), float(compactness_mean), float(concavity_mean), float(concave_points_mean), float(symmetry_mean), float(fractal_dimension_mean),
                                                        float(radius_se), float(texture_se), float(perimeter_se), float(area_se), float(smoothness_se), float(compactness_se), float(concavity_se), float(concave_points_se), float(symmetry_se), float(fractal_dimension_se),
                                                        float(radius_worst), float(texture_worst), float(perimeter_worst), float(area_worst), float(smoothness_worst), float(compactness_worst), float(concavity_worst), float(concave_points_worst), float(symmetry_worst), float(fractal_dimension_worst)]])

            if (cancer_prediction[0] == 1):
                cancer_diagnosis = 'The tumor is benign'
            else:
                cancer_diagnosis = 'The tumor is malignant'

            input_data = {
                "radius_mean": radius_mean,
                "texture_mean": texture_mean,
                "perimeter_mean": perimeter_mean,
                "area_mean": area_mean,
                "smoothness_mean": smoothness_mean,
                "compactness_mean": compactness_mean,
                "concavity_mean": concavity_mean,
                "concave_points_mean": concave_points_mean,
                "symmetry_mean": symmetry_mean,
                "fractal_dimension_mean": fractal_dimension_mean,
                "radius_se": radius_se,
                "texture_se": texture_se,
                "perimeter_se": perimeter_se,
                "area_se": area_se,
                "smoothness_se": smoothness_se,
                "compactness_se": compactness_se,
                "concavity_se": concavity_se,
                "concave_points_se": concave_points_se,
                "symmetry_se": symmetry_se,
                "fractal_dimension_se": fractal_dimension_se,
                "radius_worst": radius_worst,
                "texture_worst": texture_worst,
                "perimeter_worst": perimeter_worst,
                "area_worst": area_worst,
                "smoothness_worst": smoothness_worst,
                "compactness_worst": compactness_worst,
                "concavity_worst": concavity_worst,
                "concave_points_worst": concave_points_worst,
                "symmetry_worst": symmetry_worst,
                "fractal_dimension_worst": fractal_dimension_worst
            }
            # store the input data in the database
            breast_cancer_db.collection('breast_cancer_inputs').add(input_data)
            
            # radius_mean: mean of distances from center to points on the perimeter
            # texture_mean: standard deviation of gray-scale values
            # perimeter_mean: perimeter of tumor
            # area_mean: area of tumor
            # smoothness_mean: local variation in radius lengths
            # compactness_mean: perimeter^2 / area - 1.0
            # concavity_mean: severity of concave portions of the contour
            # concave_points_mean: number of concave portions of the contour
            # symmetry_mean: symmetry of tumor
            # fractal_dimension_mean: "coastline approximation" - 1
            # The "_se" and "_worst" versions of these attributes represent the standard error and worst (largest) value
            # of the same features, respectively. 
                        
                        
        st.success(cancer_diagnosis)


if __name__ == "__main__":
    run_ml_app()