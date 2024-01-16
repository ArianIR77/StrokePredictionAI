from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import logging
import json


app = Flask(__name__)




def process(raw_data):
    try:
        processed_columns = joblib.load('processed_columns.joblib')
        
        # One-hot encoding of categorical variables
        processed_data = pd.get_dummies(raw_data, prefix='ohe', columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
        
        # Logging processed data to a file
        with open('processed_data.json', 'w') as file:
            json.dump(processed_data.to_dict(), file, indent=4)

        # Dropping the 'id' column, if present
        if 'id' in processed_data.columns:
            processed_data = processed_data.drop("id", axis=1)
            
        if 'stroke' in processed_data.columns:
            processed_data = processed_data.drop("stroke", axis=1)
            
        if 'ohe_Other' in processed_data.columns:
            processed_data = processed_data.drop("ohe_Other", axis=1)

        # Filling missing values in 'bmi' column with the median
        processed_data['bmi'].fillna(28, inplace=True)
        
        # Ensure all columns from training are present
        for col in processed_columns:
            if col not in processed_data.columns:
                processed_data[col] = 0

        # Reorder columns to match training data
        processed_data = processed_data[processed_columns]
        
        with open('processed_data2222.json', 'w') as file:
            json.dump(processed_data.to_dict(), file, indent=4)
        
        if 'stroke' in processed_data.columns:
            processed_data = processed_data.drop("stroke", axis=1)

        # Normalizing the features
        scaler = joblib.load('scaler.joblib')
        processed_data = scaler.transform(processed_data)  

        return processed_data
    except Exception as e:
        raise Exception(f"Data preprocessing error: {e}")
    
    

def run_pipeline(data, model_path="logistic_regression_model_smote.joblib"):
    try:
        model = joblib.load(model_path)
        preprocessed_data = process(data)
        predictions = model.predict(preprocessed_data)
        return predictions.tolist()  # Convert to list if numpy array
    except Exception as e:
        raise Exception(f"Model prediction error: {e}")



@app.route('/', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        data = pd.read_csv(file_path)
        scores = run_pipeline(data)
        return jsonify({"score": scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)