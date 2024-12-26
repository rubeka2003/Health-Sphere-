import joblib
import pandas as pd

# Load the model file
model_data = joblib.load('new_diabetes_prediction_model.pkl')
rf_diabetes_model = model_data['model']
label_encoder = model_data['label_encoder']
scaler = model_data['scaler']
smoking_mapping = model_data['smoking_mapping']
feature_names = model_data['feature_names']

# Example input values
test_data = {
    'gender': 'Female',  # Use string if the label encoder expects it
    'age': 25.0,
    'hypertension': 0,
    'heart_disease': 0,
    'smoking_history': 'non-smoker',  # Use a string that matches the training labels
    'bmi': 28.5,
    'HbA1c_level': 6.6,
    'blood_glucose_level': 80.0
}

# Encode categorical variables
test_data['gender'] = label_encoder.transform([test_data['gender']])[0]
test_data['smoking_history'] = smoking_mapping[test_data['smoking_history']]

# Create a DataFrame
test_features = pd.DataFrame([[ 
    test_data['gender'],
    test_data['age'],
    test_data['hypertension'],
    test_data['heart_disease'],
    test_data['smoking_history'],
    test_data['bmi'],
    test_data['HbA1c_level'],
    test_data['blood_glucose_level']
]], columns=feature_names)

# Scale features if the scaler exists
# if scaler:
#     test_features = scaler.transform(test_features)

# Make a prediction
prediction = rf_diabetes_model.predict(test_features)[0]
probability = rf_diabetes_model.predict_proba(test_features)[0][1]

# Output the result
result = {
    'prediction': 'Positive' if prediction == 1 else 'Negative',
    'probability': f'{probability:.2%}'
}
print("Prediction Result:", result)
