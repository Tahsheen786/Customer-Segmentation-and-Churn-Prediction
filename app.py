from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the trained churn model
model = joblib.load("svm.pkl")

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the request
    data = request.json['features']  # Expecting a list of features
    data = np.array(data).reshape(1, -1)  # Reshape for single prediction

    # Predict churn using the loaded model
    prediction = model.predict(data)[0]  # Assumes binary output (0 or 1)
    
    # Respond with the prediction
    return jsonify({'prediction': int(prediction)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
