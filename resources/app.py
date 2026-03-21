from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
final_model_reloaded = joblib.load("diabetes_prediction_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    new_data = pd.DataFrame([float_features], columns=['Glucose', 'Insulin', 'BMI', 'Age'])

    prediction = final_model_reloaded.predict(new_data)

    if prediction == 1:
        pred = "Bạn có thể đang mắc bệnh tiểu đường, hãy đến thăm khám với bác sĩ."
    elif prediction == 0:
        pred = "Bạn không bị bệnh tiểu đường."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
