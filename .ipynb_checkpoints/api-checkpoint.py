from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model and scaler with error handling
try:
    model = joblib.load("fruit_classifier.joblib")
    scaler = joblib.load("scaler.joblib")
    logger.info("✅ Model and scaler loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define input schema using Pydantic
class FruitFeatures(BaseModel):
    mass: float
    width: float
    height: float
    color_score: float

# Fruit label mapping
fruit_labels = {
    1: "Apple",
    2: "Mandarin",
    3: "Orange",
    4: "Lemon"
}

@app.get("/")
def home():
    return {"message": "Fruit Classifier API is running!"}

@app.post("/predict/")
def predict_fruit(features: FruitFeatures):
    try:
        logger.debug(f"🔍 Received input: {features}")

        # Check if model and scaler are properly loaded
        if model is None or scaler is None:
            logger.error("❌ Model or scaler not loaded!")
            raise HTTPException(status_code=500, detail="Model or scaler is missing")

        # Convert input data into numpy array
        input_data = np.array([[features.mass, features.width, features.height, features.color_score]])
        logger.debug(f"🔍 Input Data (before scaling): {input_data}")

        # Scale the input data
        input_scaled = scaler.transform(input_data)
        logger.debug(f"🔍 Scaled Data: {input_scaled}")

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        logger.debug(f"🔍 Prediction Output: {prediction}")

        # Get fruit label
        fruit_name = fruit_labels.get(prediction, "Unknown")
        return {"prediction": fruit_name}
    
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        return {"error": str(e)}


