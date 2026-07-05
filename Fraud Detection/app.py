from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load pickle files
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoder.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Read input from HTML
    amount = float(request.form["amount"])
    transaction_type = request.form["type"]
    oldbalanceOrg = float(request.form["oldbalanceOrg"])
    newbalanceOrig = float(request.form["newbalanceOrig"])
    oldbalanceDest = float(request.form["oldbalanceDest"])
    newbalanceDest = float(request.form["newbalanceDest"])

    # Encode transaction type
    transaction_type = encoders["type_encoder"].transform([transaction_type])[0]

    # Create feature array
    data = np.array([[
        amount,
        transaction_type,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    ]])

    # Scale input
    data = scaler.transform(data)

    # Predict cluster
    cluster = model.predict(data)[0]

    # Display result
    if cluster == 0:
        prediction = "Potential Fraud"
    else:
        prediction = "Legitimate Transaction"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)