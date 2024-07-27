# Salary-Predictor-in-Interview--SVR
"Salary-Predictor-in-Interview-SVR" is a GitHub repository featuring a Support Vector Regression (SVR) model to predict salaries based on interview data. It includes data preprocessing, model training, and evaluation scripts for accurate salary prediction.

## API deployment


``` 
from flask_cors import CORS
from flask import Flask, jsonify
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)
# Load the model and scalers
model = joblib.load('SVM_Salary.pkl')
sc_X = joblib.load('sc_X.pkl')
sc_Y = joblib.load('sc_Y.pkl')

@app.route('/')
def home():
    return "Welcome to the SVR Model Prediction API!"

@app.route('/predict/<float:feature>', methods=['GET'])
def predict(feature):
    # Preprocess the data
    features_transformed = sc_X.transform(np.array([[feature]]))

    # Make prediction
    prediction = model.predict(features_transformed)

    # Inverse transform to original scale
    prediction_original_scale = sc_Y.inverse_transform(prediction.reshape(-1, 1))

    # Return the result as a JSON response
    return jsonify({'prediction': prediction_original_scale[0, 0]})

if __name__ == '__main__':
    app.run(debug=True)

```

<be><br>
![image](https://github.com/user-attachments/assets/4779c77f-4e76-44f5-a475-d7f72ff7b823)<br>
![image](https://github.com/user-attachments/assets/e108512d-a9f3-434a-b67a-c3cf1b69f248)


## Angular Application

![image](https://github.com/user-attachments/assets/08eababb-950a-44d6-b005-cadc173d1a90)

## Install flask dependencies


### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   python -m venv venv
   pip install -r requirements.txt
   pip install Flask flask_cors joblib numpy

   ```






