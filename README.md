# Salary-Predictor-in-Interview--SVR
"Salary-Predictor-in-Interview-SVR" is a GitHub repository featuring a Support Vector Regression (SVR) model to predict salaries based on interview data. It includes data preprocessing, model training, and evaluation scripts for accurate salary prediction.

## API deployment


``` 
from flask import Flask, request, jsonify
import joblib
import numpy as np
import sklearn

app = Flask(__name__)

model = joblib.load('SVM_Salary.pkl')
sc_X = joblib.load('sc_X.pkl')
sc_Y = joblib.load('sc_Y.pkl')

@app.route('/')

def home():
    return "Welcome to the SVR Model Prediction API!"

@app.route('/predict', methods=['POST'])

def predict():
    # Get the data from the POST request
    data = request.json
    print("Received data:", data)
    features = np.array([[data['feature']]])

    # Preprocess the data
    features_transformed = sc_X.transform(features)
 # Make prediction
    prediction = model.predict(features_transformed)

    # Inverse transform to original scale
    prediction_original_scale = sc_Y.inverse_transform(prediction.reshape(-1, 1))

    # Return the result as a JSON response
    return jsonify({'prediction': prediction_original_scale[0, 0]})

if __name__ == '__main__':
    app.run(debug=True)
```
