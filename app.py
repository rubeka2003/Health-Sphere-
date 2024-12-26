from flask import Flask, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load models
model_data = joblib.load(open('new_diabetes_prediction_model.pkl', 'rb'))

diabetes_model = model_data['model']
label_encoder = model_data['label_encoder']
scaler = model_data['scaler']
feature_names = model_data['feature_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    prediction_prob = None
    
    if request.method == 'POST':
        try:
            # Collect form inputs
            gender = request.form['gender']
            age = float(request.form['age'])
            hypertension = float(request.form['hypertension'])
            heart_disease = float(request.form['heart_disease'])
            smoking_history = request.form['smoking_history']
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])

            # Encode categorical variables
            gender_encoded = label_encoder['gender'].transform([gender])[0]
            smoking_encoded = label_encoder['smoking_history'].transform([smoking_history])[0]

            # Prepare features
            features = np.array([[
                gender_encoded, age, hypertension, heart_disease, 
                smoking_encoded, bmi, HbA1c_level, blood_glucose_level
            ]])

            # Scale features
            features_scaled = scaler.transform(features)

            # Make predictions
            prediction = diabetes_model.predict(features_scaled)[0]
            prediction_prob = diabetes_model.predict_proba(features_scaled)[0]

            # Determine result
            if prediction == 1:
                result = f"Diabetes Detected (Probability: {prediction_prob[1]:.2%})"
            else:
                result = f"No Diabetes Detected (Probability: {prediction_prob[0]:.2%})"

        except Exception as e:
            result = f"Error processing prediction: {str(e)}"

    return render_template('diabetes.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)