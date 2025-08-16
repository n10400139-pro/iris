# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import joblib
import numpy as np

app = FastAPI(title="Iris Species Prediction API")

# Load model and dataset when app starts
@app.on_event("startup")
def load_objects():
    global model, iris
    model = joblib.load("model.pkl")
    iris = joblib.load("iris_dataset.pkl")

# Input validation: expects exactly 4 floats for features
class PredictionInput(BaseModel):
    features: conlist(float, min_items=4, max_items=4) # type: ignore

# Output schema with predicted species and confidence score
class PredictionOutput(BaseModel):
    predicted_species: str
    confidence: float = None

@app.get("/", summary="Health check")
def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.array(input_data.features).reshape(1, -1)
        pred_idx = model.predict(features)[0]
        pred_proba = model.predict_proba(features).max()
        species = iris.target_names[pred_idx]
        return PredictionOutput(predicted_species=species, confidence=float(pred_proba))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "Logistic Regression",
        "problem_type": "classification",
        "features": iris.feature_names,
        "target_names": iris.target_names.tolist()
    }
