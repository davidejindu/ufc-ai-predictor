# UFC Fight Prediction ML Pipeline

A complete machine learning pipeline for predicting UFC fight outcomes using XGBoost and FastAPI.

## Features
- Scraped 21,000+ UFC fights and 3,600+ fighters
- Advanced feature engineering (Elo ratings, win streaks, recency)
- XGBoost model with 0.61 AUC
- FastAPI microservice with real-time predictions
- SHAP-based explainability

## Tech Stack
- Python, pandas, scikit-learn, XGBoost
- FastAPI, uvicorn
- SHAP for model interpretability

## Usage
```bash
# Start the API
python serve_api.py

# Make prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'
```

## API Response
```json
{
  "probability": 0.636,
  "top_3_shap": [...]
}
```
```
