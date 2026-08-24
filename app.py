from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    error_text = None

    if request.method == "POST":
        try:
            # Extract inputs from HTML form
            features = [
                float(request.form["area"]),
                float(request.form["bedrooms"]),
                float(request.form["bathrooms"]),
                float(request.form["house_age"]),
                float(request.form["distance_to_city"]),
                float(request.form["parking"]),
                float(request.form["floor"]),
                float(request.form["nearby_schools"]),
                float(request.form["crime_rate"])
            ]
            
            # Scale and predict
            scaled_features = scaler.transform([features])
            prediction = model.predict(scaled_features)[0]
            
            # Format output in Indian Rupees
            prediction_text = f"₹ {round(prediction, 2):,}"
        except Exception as e:
            error_text = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction_text, error=error_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)