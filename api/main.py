import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/final_model.joblib')
COLUMNS_PATH = os.path.join(os.path.dirname(__file__), '../models/expected_columns.joblib')

model = None
expected_columns = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, expected_columns
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        model = joblib.load(MODEL_PATH)
        expected_columns = joblib.load(COLUMNS_PATH)
        print("Modele charge avec succes.")
    else:
        print("ATTENTION : Le modele ou les colonnes attendues n'ont pas ete trouves. Veuillez executer src/train.py d'abord.")
    yield
    model = None
    expected_columns = None

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0",
    description="API de prediction du risque de resiliation client (churn). Fournit une probabilite de churn et une prediction binaire a partir des caracteristiques d'un client.",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    features: dict

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "age": 35,
                    "tenure_months": 12,
                    "monthly_fee": 50,
                    "total_revenue": 600,
                    "support_tickets": 2,
                    "csat_score": 3.5,
                    "contract_type": "Monthly",
                    "payment_failures": 1,
                    "usage_growth_rate": -0.1
                }
            }
        }
    }

@app.get("/health", summary="Verification de l'etat du service")
def health_check():
    """Verifie si l'API est active et si le modele est correctement charge."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge. Verifiez que src/train.py a ete execute.")
    return {"status": "ok", "model_loaded": True}

@app.get("/model-info", summary="Informations sur le modele en production")
def model_info():
    """Retourne des informations sur le modele charge."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge.")
    return {
        "model_type": type(model.named_steps['classifier']).__name__,
        "n_features": len(expected_columns),
        "pipeline_steps": list(model.named_steps.keys())
    }

@app.post("/predict", summary="Prediction du risque de churn")
def predict(request: PredictionRequest):
    """
    Recoit les caracteristiques d'un client sous forme de dictionnaire JSON
    et retourne :
    - **prediction** : 0 (client fidele) ou 1 (risque de churn)
    - **churn_probability** : probabilite de resiliation entre 0 et 1
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge.")

    if not request.features:
        raise HTTPException(status_code=422, detail="Le dictionnaire 'features' ne peut pas etre vide.")

    try:
        input_data = pd.DataFrame([request.features])
        input_encoded = pd.get_dummies(input_data)

        input_aligned = pd.DataFrame(0, index=[0], columns=expected_columns)
        for col in expected_columns:
            if col in input_encoded.columns:
                input_aligned[col] = input_encoded[col].values

        prediction = model.predict(input_aligned)[0]
        probability = model.predict_proba(input_aligned)[0][1]

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 4),
            "risk_level": "eleve" if probability > 0.5 else "faible"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prediction : {str(e)}")
