from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


# Load Trained Models

with open("model/heart_model.pkl", "rb") as file:
    heart_model = pickle.load(file)

with open("model/diabetes_model.pkl", "rb") as file:
    diabetes_model = pickle.load(file)


# Home Page

@app.route("/")
def home():
    return render_template("index.html")


# Heart Page

@app.route("/heart")
def heart():
    return render_template("heart.html")


# Diabetes Page

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


# Heart Prediction

@app.route("/predict_heart", methods=["POST"])
def predict_heart():

    age = float(request.form["age"])
    sex = float(request.form["sex"])
    chest_pain_type = float(request.form["chest_pain_type"])
    resting_bps = float(request.form["resting_bps"])
    cholesterol = float(request.form["cholesterol"])
    fasting_blood_sugar = float(request.form["fasting_blood_sugar"])
    resting_ecg = float(request.form["resting_ecg"])
    max_heart_rate = float(request.form["max_heart_rate"])
    exercise_angina = float(request.form["exercise_angina"])
    oldpeak = float(request.form["oldpeak"])
    st_slope = float(request.form["st_slope"])

    prediction = heart_model.predict([[age,
                                       sex,
                                       chest_pain_type,
                                       resting_bps,
                                       cholesterol,
                                       fasting_blood_sugar,
                                       resting_ecg,
                                       max_heart_rate,
                                       exercise_angina,
                                       oldpeak,
                                       st_slope]])

    if prediction[0] == 1:
        result = "❤️ Heart Disease Detected"
    else:
        result = "✅ No Heart Disease"

    return render_template("result.html", result=result)


# Diabetes Prediction

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():

    pregnancies = float(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    bloodpressure = float(request.form["bloodpressure"])
    skinthickness = float(request.form["skinthickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    dpf = float(request.form["dpf"])
    age = float(request.form["age"])

    prediction = diabetes_model.predict([[pregnancies,
                                          glucose,
                                          bloodpressure,
                                          skinthickness,
                                          insulin,
                                          bmi,
                                          dpf,
                                          age]])

    if prediction[0] == 1:
        result = "🍬 Diabetes Detected"
    else:
        result = "✅ No Diabetes"

    return render_template("result.html", result=result)


# Run Flask App

if __name__ == "__main__":
    app.run(debug=True)