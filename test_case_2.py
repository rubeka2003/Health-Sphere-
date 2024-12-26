from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd


app = Flask(__name__)

# Initialize label encoders
# label_encoders = {
#     'gender': LabelEncoder(),
#     'smoking_history': LabelEncoder()
# }

# label_encoders['gender'].fit(['Male', 'Female'])
# label_encoders['smoking_history'].fit(['never', 'former', 'current', 'not current', 'ever'])

# Load the random forest model
model_data = joblib.load('new_diabetes_prediction_model.pkl')
rf_diabetes_model = model_data['model']
label_encoder = model_data['label_encoder']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    if request.method == 'POST':

            # Get values from the form
            gender = request.form['gender']
            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            smoking_history = int(request.form['smoking_history'])
            bmi = float(request.form['bmi'])
            hba1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])
            
            
            #smoking_category = smoking_mapping[smoking_history]


            

            gender_encoded = label_encoder.transform([gender])[0]
            #smoking_encoded = label_encoder.transform([smoking_category])[0]

            
            # Create feature array
            # features = np.array([[
            #     gender_encoded, age, hypertension, heart_disease,
            #     smoking_history, bmi, hba1c_level, blood_glucose_level
            # ]])

            features = pd.DataFrame([[
                gender_encoded, age, hypertension, heart_disease,
                smoking_history, bmi, hba1c_level, blood_glucose_level
            ]], columns=feature_names)
            
            # Scale features
            # Make prediction
            prediction = rf_diabetes_model.predict(features)[0]
            probability = rf_diabetes_model.predict_proba(features)[0][1]
            
            # result = {
            #     'prediction': 'Positive' if prediction == 1 else 'Negative',
            #     'probability': f'{probability:.2%}'
            # }

            if prediction == 1:
                result = result = f"""
                Status: High Risk
                Message: Diabetes Risk Detected
                
                Recommendations:
                - Consult a healthcare provider
                - Monitor blood glucose regularly 
                - Maintain a healthy diet
                - Exercise regularly
                """
            else:
                result = result = f"""
                Status: Low Risk
                Message: No Significant Diabetes Risk
                
                Recommendations:
                - Maintain healthy lifestyle
                - Regular health checkups
                - Balanced diet
                - Stay active
                """
            
    return render_template('diabetes.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)