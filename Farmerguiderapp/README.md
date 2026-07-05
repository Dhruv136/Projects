# Farmer Guiding App 🌾

## Project Overview

The Farmer Guiding App is a Machine Learning and Flask-based web application that helps farmers make better agricultural decisions.

The application provides:

* 🌱 Crop Prediction
* 🌿 Fertilizer Prediction

Users enter soil and environmental details through a web interface, and the trained machine learning models predict the most suitable crop or fertilizer.

## Technologies Used

* Python
* Flask
* HTML
* CSS
* NumPy
* Pandas
* Scikit-learn
* Pickle

## Features

* Crop prediction using soil and weather parameters.
* Fertilizer recommendation based on input data.
* Simple and user-friendly web interface.
* Machine learning model integration with Flask.

## Project Structure

```text
Farmer_Guiding_App/
│
├── app.py
├── model/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── templates/
│   ├── index.html
│   ├── crop.html
│   └── fertilizer.html
│
├── static/
│   └── style.css
│
└── README.md
```

## Input Parameters

### Crop Prediction

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall

### Fertilizer Prediction

* Soil and crop-related parameters as required by the fertilizer prediction model.

## How to Run

1. Clone or download the project.
2. Install the required libraries.
3. Place the trained model files inside the `model` folder.
4. Run the Flask application:

```bash
python app.py
```

5. Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## Output

* Predicts the most suitable crop.
* Recommends the appropriate fertilizer.

## Future Improvements

* Weather API integration.
* Disease detection using images.
* Multi-language support.
* Farmer dashboard with crop history.

## Developed By

Dhruv Kokcha
