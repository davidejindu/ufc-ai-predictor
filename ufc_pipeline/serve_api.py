import pickle
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shap
import traceback
import os
import joblib
import pandas as pd

# Load model, imputer, and feature list
with open('ufc_model.pkl', 'rb') as f:
    model = pickle.load(f)
print('Loaded model type:', type(model))
print("CWD:", os.getcwd())
print("ufc_scaler.pkl exists:", os.path.exists("ufc_scaler.pkl"))
imputer_path = os.path.abspath('ufc_scaler.pkl')
print(f"Loading imputer from: {imputer_path}")
imputer = joblib.load(imputer_path)
print(f"Loaded imputer type: {type(imputer)}")
with open('feature_list.json', 'r') as f:
    feature_list = json.load(f)

# SHAP explainer (re-create at startup)
explainer = shap.TreeExplainer(model)

app = FastAPI(title="UFC Fight Predictor API")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print("Exception occurred:", exc)
    traceback.print_exc()
    return HTTPException(status_code=500, detail=str(exc))

@app.get("/ping")
def ping():
    print("Ping received")
    return {"ping": "pong"}

class PredictRequest(BaseModel):
    features: dict

@app.post("/predict")
def predict(req: PredictRequest):
    print("Received /predict request")
    # Ensure all required features are present
    features = req.features
    feature_names = list(imputer.feature_names_in_)
    X = pd.DataFrame([[features.get(f, np.nan) for f in feature_names]], columns=feature_names)
    print("DataFrame columns:", X.columns.tolist())
    print("DataFrame shape:", X.shape)
    print("Imputing values")
    X_imp = imputer.transform(X)
    print("Predicting probability")
    prob = float(model.predict_proba(X_imp)[0, 1])
    print("Calculating SHAP values")
    shap_vals = explainer.shap_values(X_imp)[0]
    print("Selecting top-3 SHAP features")
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_3 = [
        {
            "feature": feature_list[i],
            "value": X_imp[0, i],
            "shap_value": float(shap_vals[i])
        }
        for i in top_idx
    ]
    print("Returning response")
    return {"probability": prob, "top_3_shap": top_3} 