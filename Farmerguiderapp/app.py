from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load  crop model 

crop_model = pickle.load(open("model/best_model.pkl", "rb"))
crop_scaler = pickle.load(open("model/scaler.pkl", "rb"))
crop_encoder = pickle.load(open("model/encoder.pkl", "rb"))
# load fertilizer model

fert_model = pickle.load(open("model/best_model2.pkl", "rb"))
fert_scaler = pickle.load(open("model/scaler2.pkl", "rb"))
fert_encoders = pickle.load(open("model/encoders2.pkl", "rb"))
soil_encoder = fert_encoders["soil_encoder"]
fcrop_encoder = fert_encoders["crop_encoder"]
fert_encoder = fert_encoders["encoder"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/crop", methods=["GET", "POST"])
def crop():

    if request.method == "POST":

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Scale the input
        data = crop_scaler.transform(data)

        # Predict
        # Predict (model outputs an encoded label, e.g. 5)
        prediction = crop_model.predict(data)

        # Convert encoded label back to crop name (e.g. "rice")
        crop_name = crop_encoder.inverse_transform(prediction)[0]

        return render_template("crop.html", prediction=crop_name)

    return render_template("crop.html")
@app.route("/fertilizer", methods=["GET", "POST"])
def fertilizer():

    if request.method == "POST":

        temperature = float(request.form["temperature"])
        moisture = float(request.form["moisture"])
        rainfall = float(request.form["rainfall"])
        ph = float(request.form["ph"])
        nitrogen = float(request.form["nitrogen"])
        phosphorous = float(request.form["phosphorous"])
        potassium = float(request.form["potassium"])
        carbon = float(request.form["carbon"])
        soil = soil_encoder.transform([request.form["soil"]])[0]
        crop = fcrop_encoder.transform([request.form["crop"]])[0]

        data = np.array([[temperature, moisture, rainfall, ph, nitrogen,
                           phosphorous, potassium, carbon, soil, crop]])

        data = fert_scaler.transform(data)
        prediction = fert_model.predict(data)
        fert_name = fert_encoder.inverse_transform(prediction)[0]

        return render_template(
            "fertilizer.html",
            prediction=fert_name,
            soils=soil_encoder.classes_,
            crops=fcrop_encoder.classes_,
        )

    return render_template(
        "fertilizer.html",
        soils=soil_encoder.classes_,
        crops=fcrop_encoder.classes_,
    )


if __name__ == "__main__":
    app.run(debug=True)
