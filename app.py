import flask
from flask import request
from flask_cors import CORS
import json
import joblib

app = flask.Flask(__name__)

CORS(app)

# main index page (root route)
@app.route("/")
def home():
    return "<h1>Salary Prediction API</h1><p>BAIS:3300 - Digital Product Development</p><p>Weston Rauschuber</p>"

# predict route
@app.route("/predict", methods=["POST"])
def predict():
    print("inside predict")
    
    # load the model
    model = joblib.load("salary_predict_model.ml")

    # get values from json
    prediction_variables = request.get_json()
    
    print(prediction_variables)
    
    # store the json values into python variables
    age = prediction_variables["age"]
    gender = prediction_variables["gender"]
    country = prediction_variables["country"]
    highest_deg = prediction_variables["highest_deg"]
    coding_exp = prediction_variables["coding_exp"]
    title = prediction_variables["title"]
    company_size = prediction_variables["company_size"]

    print(age, gender, country, highest_deg, coding_exp, title, company_size)

    # make a prediction using the python variables
    # ensure the variables are in the same order as the model was trained on
    salary_prediction = model.predict(
        [
            [
                int(age),
                int(gender),
                int(country),
                int(highest_deg),
                int(coding_exp),
                int(title),
                int(company_size),
            ]
        ]
    )
    
    print(salary_prediction)

    # convert NumPy array to a Python list and extract the first value
    salary_prediction_value = salary_prediction.tolist()[0]

    # return the result as a JSON response
    return json.dumps({"predicted_salary": salary_prediction_value})