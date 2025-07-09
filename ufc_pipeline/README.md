# UFC Fight Prediction ML Pipeline

A complete machine learning pipeline for predicting UFC fight outcomes using advanced feature engineering, XGBoost, and FastAPI.

## 🏆 Project Overview

This project demonstrates end-to-end ML development: from data scraping and feature engineering to model training and production API deployment. The pipeline processes 21,000+ UFC fights and 3,600+ fighters to predict fight outcomes with explainable AI insights.

## 🛠️ Tech Stack

- **Data Processing**: Python, pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **API Framework**: FastAPI, uvicorn
- **Model Interpretability**: SHAP
- **Data Storage**: CSV files

## 📊 Accomplishments

### 1. Data Engineering & Feature Creation

- **Scraped, cleaned, and engineered features** from 21,000+ UFC fights and 3,600+ fighters
- **Implemented advanced feature engineering**, including:
  - Dynamic win rates and streaks
  - Fight recency metrics
  - Elo rating system for fighter rankings
  - Height/weight/reach differentials
  - Finish rates (KO, submission, decision)

### 2. Model Development

- **Trained and tuned an XGBoost classifier** for fight outcome prediction
- **Achieved an AUC of ~0.61** on test data (as of model v1)
- **Performed SHAP analysis** to interpret model predictions and identify key features
- **Feature importance insights**: Opponent finish rate, Elo ratings, and total fights differential are top predictors

### 3. Production-Ready ML Pipeline

- **Built a robust Python pipeline** for data preprocessing, feature engineering, model training, and evaluation
- **Used SimpleImputer** for missing value handling
- **Ensured feature consistency** between training and inference
- **Automated model persistence** with joblib for easy deployment

### 4. FastAPI ML Microservice

- **Developed a FastAPI application** to serve real-time fight outcome predictions and SHAP explanations
- **Implemented a `/predict` endpoint** that:
  - Accepts JSON payloads with all required features
  - Handles missing values and feature ordering robustly
  - Returns both prediction probabilities and top SHAP feature attributions

### 5. Debugging & Robustness

- **Diagnosed and resolved complex issues** with imputer/model file mismatches and feature shape errors
- **Automated feature alignment** between the imputer, model, and API
- **Added detailed logging and debug output** for easier maintenance and troubleshooting
- **Implemented comprehensive error handling** for production reliability

### 6. API Integration Readiness

- **Validated the API** with realistic curl/Postman requests
- **Documented feature requirements** and API specifications
- **Ready for integration** with frontend applications or other services

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn shap joblib
```

### Training the Model

```bash
python ufc_modeling.py
```

### Starting the API Server

```bash
python serve_api.py
```

The API will be available at `http://localhost:8000` for local testing.

### Making Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "f_height": 72,
      "o_height": 70,
      "f_weight": 170,
      "o_weight": 170,
      "f_win_rate": 0.7,
      "o_win_rate": 0.6,
      "f_finish_rate": 0.5,
      "o_finish_rate": 0.4,
      "f_ko_rate": 0.2,
      "o_ko_rate": 0.1,
      "f_sub_rate": 0.2,
      "o_sub_rate": 0.1,
      "f_total_fights": 20,
      "o_total_fights": 18,
      "height_diff": 2,
      "weight_diff": 0,
      "win_rate_diff": 0.1,
      "finish_rate_diff": 0.1,
      "ko_rate_diff": 0.1,
      "sub_rate_diff": 0.1,
      "total_fights_diff": 2,
      "days_since_last_fight": 120,
      "is_title_fight": 0,
      "five_fight_win_streak": 1,
      "elo_diff": 50,
      "f_elo": 1550,
      "o_elo": 1500
    }
  }'
```

## 📈 API Response Format

```json
{
  "probability": 0.6365063190460205,
  "top_3_shap": [
    {
      "feature": "o_elo",
      "value": 1500.0,
      "shap_value": -0.4557090997695923
    },
    {
      "feature": "o_sub_rate",
      "value": 0.1,
      "shap_value": 0.32399091124534607
    },
    {
      "feature": "o_finish_rate",
      "value": 0.4,
      "shap_value": 0.2687806189060211
    }
  ]
}
```

## 📁 Project Structure

```
ufc_pipeline/
├── ufc_modeling.py          # Main modeling script
├── serve_api.py             # FastAPI server
├── feature_list.json        # Required features
├── ufc_model.pkl           # Trained XGBoost model (binary artifacts like this are usually stored in /models/ or a storage bucket, not committed to source)
├── ufc_scaler.pkl          # Fitted imputer (see above)
├── fighters.csv            # Fighter data
├── fights.csv              # Fight data
└── README.md               # This file
```

## 🔧 Key Features

- **27 engineered features** including fighter stats, differentials, and historical performance
- **Real-time predictions** with probability scores
- **SHAP explanations** for model interpretability
- **Robust error handling** for production use
- **Comprehensive logging** for debugging and monitoring

## 📊 Model Performance

- **AUC Score**: 0.604 (as of model v1)
- **Accuracy**: 54.2% (as of model v1)
- **Dataset Size**: 18,283 fights
- **Unique Fighters**: 3,535

## ⚠️ Data & Security Notes

- **Dataset licensing**: Scraped data is for personal/educational use; check source site’s TOS before distributing.
- **Secrets**: Store credentials (API keys, etc.) in environment variables or a secrets manager. **Never commit secrets to source control.**

## 🤝 Contributing

This project demonstrates end-to-end ML development skills. Feel free to fork and extend with additional features like:

- More advanced feature engineering
- Different ML algorithms
- Web interface
- Real-time data updates

## 📝 License

This project is for educational and portfolio purposes.

---

**Built with ❤️ for UFC fight prediction and ML engineering demonstration**
