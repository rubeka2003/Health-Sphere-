# from flask import Flask, render_template, request
# import pickle
# import os
# import numpy as np

# app = Flask(__name__)

# # Load the random forest model
# try:
#     model_data = pickle.load(open('random_forest_diabetes_model.pkl', 'rb'))
#     rf_diabetes_model = model_data['model']
#     label_encoders = model_data['label_encoders']
#     scaler = model_data['scaler']
#     feature_names = model_data['feature_names']
# except:
#     print("Model loading initiated")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/diabetes', methods=['GET', 'POST'])
# def diabetes():
#     if request.method == 'POST':
#         try:
#             # Get input features from form
#             gender = request.form['gender']
#             age = float(request.form['age'])
#             hypertension = float(request.form['hypertension'])
#             heart_disease = float(request.form['heart_disease'])
#             smoking_history = request.form['smoking_history']
#             bmi = float(request.form['bmi'])
#             HbA1c_level = float(request.form['HbA1c_level'])
#             blood_glucose_level = float(request.form['blood_glucose_level'])

#             # Encode categorical variables
#             gender_encoded = label_encoders['gender'].transform([gender])[0]
#             smoking_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]

#             # Create feature array
#             features = np.array([[
#                 gender_encoded, age, hypertension, heart_disease, 
#                 smoking_encoded, bmi, HbA1c_level, blood_glucose_level
#             ]])

#             # Scale features
#             features_scaled = scaler.transform(features)

#             # Make prediction
#             prediction = rf_diabetes_model.predict(features_scaled)[0]
#             prediction_prob = rf_diabetes_model.predict_proba(features_scaled)[0]

#             result = {
#                 'prediction': int(prediction),
#                 'probability': float(prediction_prob[1])
#             }

#             return result

#         except Exception as e:
#             return {'error': str(e)}
#         return render_template('diabetes.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Initialize label encoders
label_encoders = {
    'gender': LabelEncoder(),
    'smoking_history': LabelEncoder()
}

label_encoders['gender'].fit(['Male', 'Female'])
label_encoders['smoking_history'].fit(['never', 'former', 'current', 'not current', 'ever'])

# Load the random forest model
try:
    model_data = joblib.load(open('new_diabetes_prediction_model.pkl', 'rb'))
    rf_diabetes_model = model_data['model']
    label_encoders = model_data['label_encoders']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
except:
    print("Model loading initiated")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    if request.method == 'POST':
        try:
            # Get input features from form
            gender = request.form['gender']
            age = float(request.form['age'])
            hypertension = float(request.form['hypertension'])
            heart_disease = float(request.form['heart_disease'])
            smoking_history = request.form['smoking_history']
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])

            # Encode categorical variables
            gender_encoded = label_encoders['gender'].transform([gender])[0]
            smoking_encoded = label_encoders['smoking_history'].transform([smoking_history])[0]

            # Create feature array
            features = np.array([[
                gender_encoded, age, hypertension, heart_disease, 
                smoking_encoded, bmi, HbA1c_level, blood_glucose_level
            ]])

            # Scale features
            features_scaled = scaler.transform(features)

            # Make prediction
            prediction = rf_diabetes_model.predict(features_scaled)[0]
            prediction_prob = rf_diabetes_model.predict_proba(features_scaled)[0]

            result = {
                'prediction': int(prediction),
                'probability': float(prediction_prob[1]),
                'status': 'success'
            }
            
            #return render_template('diabetes.html', result=result)

        except Exception as e:
            return jsonify({'error': str(e), 'status': 'error'})
    
    return render_template('diabetes.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

