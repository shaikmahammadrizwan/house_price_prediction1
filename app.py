from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib


# Create FastAPI application
app = FastAPI()


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model and scaler
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")


# Input data structure
class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    house_age: float
    distance_to_city: float
    parking: int
    floor: int
    nearby_schools: int
    crime_rate: float


# Home route
@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


# Prediction route
@app.post("/predict")
def predict(data: HouseData):

    new_house = [[
        data.area,
        data.bedrooms,
        data.bathrooms,
        data.house_age,
        data.distance_to_city,
        data.parking,
        data.floor,
        data.nearby_schools,
        data.crime_rate
    ]]

    new_house_scaled = scaler.transform(new_house)

    prediction = model.predict(new_house_scaled)

    return {
        "predicted_house_price": float(prediction[0])
    }