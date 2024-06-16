from typing import Dict, Any
from config import Config
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from loguru import logger
import json

CONF = Config()
model = joblib.load(f"{CONF.paths.TRAIN_MODEL}/{CONF.model.MODEL_NAME}.pkl")
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict(data: Dict[str, Any]):
    """
    Endpoint to make predictions using a trained model.

    Parameters:
    - data: JSON payload containing the input data for prediction.

    Returns:
    - dict: JSON response containing the prediction result.

    Example JSON Input:\n
    ```
    {
        "Name": "Chris Webber",
        "GP": 76,
        "MIN": 32.1,
        "PTS": 17.5,
        "FGM": 7.5,
        "FGA": 13.6,
        "FG%": 55.2,
        "3P Made": 0.0,
        "3PA": 0.2,
        "3P%": 0.0,
        "FTM": 2.5,
        "FTA": 4.7,
        "FT%": 53.2,
        "OREB": 4.0,
        "DREB": 5.1,
        "REB": 9.1,
        "AST": 3.6,
        "STL": 1.2,
        "BLK": 2.2,
        "TOV": 2.7
    }
    ```
    """
    try:

        input_data = pd.DataFrame.from_dict([data])
        logger.info("Input data to frame.")

        prediction_prob = model.predict_proba(input_data)[:, 1]
        prediction = (prediction_prob >= 0.35).astype(int)
        logger.info(
            f"Predicted successfully: {prediction_prob} - {prediction}"
        )

        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )
