from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and encoders once
model = joblib.load('model/loan_approval_model.pkl')
label_encoders = joblib.load('model/label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Please submit the form instead of accessing this page directly."
    # Get form data
    data = request.form.to_dict()

    # Convert numeric fields to proper types
    data['ApplicantIncome'] = float(data['ApplicantIncome'])
    data['CoapplicantIncome'] = float(data['CoapplicantIncome'])
    data['LoanAmount'] = float(data['LoanAmount'])
    data['Loan_Amount_Term'] = float(data['Loan_Amount_Term'])
    data['Credit_History'] = float(data['Credit_History'])

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Encode categorical columns using saved label encoders
    for col in input_df.select_dtypes(include='object').columns:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Predict
    pred = model.predict(input_df)[0]
    result = 'Approved' if pred == 1 else 'Rejected'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 locally
    app.run(host="0.0.0.0", port=port)
