from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load Tuned Model
tuned_model = joblib.load('tuned_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    age = int(request.form['age'])
    job = int(request.form['job'])
    marital = int(request.form['marital'])
    education = int(request.form['education'])
    is_default = int(request.form['isDefault'])
    has_housing_loan = int(request.form['hasHousingLoan'])
    has_personal_loan = int(request.form['hasPersonalLoan'])
    contact = int(request.form['contact'])
    month = int(request.form['month'])
    day_of_week = int(request.form['dayOfWeek'])
    duration = int(request.form['duration'])
    campaign = int(request.form['campaign'])
    pdays = int(request.form['pdays'])
    previous = int(request.form['previous'])
    previous_attempt = int(request.form['previousAttempt'])

    # Create DataFrame with user input
    new_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [is_default],
        'housing': [has_housing_loan],
        'loan': [has_personal_loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [previous_attempt]
    })

    
    prediction = tuned_model.predict(new_data)

    prediction_label = 'Yes' if prediction[0] == 1 else 'NO'

    prediction = int(prediction[0])

    # Return prediction as JSON response
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
