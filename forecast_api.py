from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import os
import torch
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from datetime import datetime
from ollama import Client
import asyncio

# Set CPU-only for PyTorch
torch.set_num_threads(2)  # Restrict threads to optimize performance on Raspberry Pi

# Constants and Environment Variables
MODEL_DIR = "model_checkpoints"
LATEST_MODEL = "latest_model.pth"

os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI()

# Define request data models
class DataPoint(BaseModel):
    ds: str
    y: float
    unique_id: str

class RequestData(BaseModel):
    data: List[DataPoint]

class LLMRequest(BaseModel):
    prompt: str
    model: str = "tinyllama"
    temperature: float = 0.7
    max_tokens: int = 500

class EvaluationRequest(BaseModel):
    actual: List[DataPoint]
    predictions: List[DataPoint]
# Global variables
global_nn_model = None
last_predictions = {}

# Ollama client setup
try:
    client = Client(host="http://localhost:11434")
except Exception as e:
    print(f"Ollama client error: {str(e)}")
# Load the latest model
def load_model():
    global global_nn_model
    try:
        if os.path.exists(os.path.join(MODEL_DIR, LATEST_MODEL)):
            model = LSTM(h=1, input_size=24, max_steps=50, batch_size=8)  # Reduce batch size
            global_nn_model = NeuralForecast(models=[model], freq="5s")
            state_dict = torch.load(os.path.join(MODEL_DIR, LATEST_MODEL), map_location="cpu")
            global_nn_model.models[0].load_state_dict(state_dict)
            print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/train")
async def train_model(request_data: RequestData):
    global global_nn_model
    try:
        df = pd.DataFrame([dp.dict() for dp in request_data.data])
        df["ds"] = pd.to_datetime(df["ds"])
        if len(df) < 24:
            raise HTTPException(422, "Insufficient data points.")

        # Train model
        model = LSTM(h=1, input_size=24, max_steps=50, batch_size=8)
        global_nn_model = NeuralForecast(models=[model], freq="60s")
        global_nn_model.fit(df)

        # Save model
        torch.save(global_nn_model.models[0].state_dict(), os.path.join(MODEL_DIR, LATEST_MODEL))
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(500, f"Training error: {str(e)}")



@app.post("/predict")
async def predict_neural(request_data: RequestData):
    global global_nn_model
    if global_nn_model is None:
        load_model()
    if global_nn_model is None:
        raise HTTPException(400, "Model not trained yet.")

    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([dp.dict() for dp in request_data.data])
        df["ds"] = pd.to_datetime(df["ds"])

        # Debug input data
        print("Input DataFrame for prediction:")
        print(df)

        # Perform prediction
        forecast = global_nn_model.predict(df)

        # Debug raw forecast output
        print("Raw forecast output:")
        print(forecast)

        # Serialize the 'ds' column to ISO 8601 string format
        forecast_result = forecast.reset_index()
        forecast_result["ds"] = forecast_result["ds"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # Convert Timestamp to string
        forecast_result = forecast_result.to_dict(orient="records")

        # Debug processed forecast output
        print("Processed forecast result:")
        print(forecast_result)

        return {"predictions": forecast_result}
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.post("/evaluate")
async def evaluate_model(evaluation_request: EvaluationRequest):
    try:
        # Convert input data to DataFrames
        actual_df = pd.DataFrame([dp.dict() for dp in evaluation_request.actual])
        predictions_df = pd.DataFrame([dp.dict() for dp in evaluation_request.predictions])

        # Ensure the 'ds' column is datetime for both
        actual_df["ds"] = pd.to_datetime(actual_df["ds"])
        predictions_df["ds"] = pd.to_datetime(predictions_df["ds"])

        # Sort DataFrames by 'ds' (required for merge_asof)
        actual_df = actual_df.sort_values("ds")
        predictions_df = predictions_df.sort_values("ds")

        # Merge with tolerance of 5 seconds
        merged_df = pd.merge_asof(
            actual_df, predictions_df, on="ds", direction="nearest", tolerance=pd.Timedelta("5s"), suffixes=("_actual", "_pred")
        )

        # Drop rows with invalid float values
        merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_actual", "y_pred"])

        if merged_df.empty:
            raise HTTPException(400, "No valid data points available for evaluation after cleaning and merging.")

        # Calculate evaluation metrics
        mae = np.mean(np.abs(merged_df["y_actual"] - merged_df["y_pred"]))
        mse = np.mean((merged_df["y_actual"] - merged_df["y_pred"]) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((merged_df["y_actual"] - merged_df["y_pred"]) / merged_df["y_actual"])) * 100

        # Return the calculated metrics
        return {
            "mae": round(mae, 3),
            "mse": round(mse, 3),
            "rmse": round(rmse, 3),
            "mape": round(mape, 3),
            "message": "Evaluation metrics calculated successfully"
        }

    except ValueError as ve:
        raise HTTPException(400, f"Invalid float values detected: {str(ve)}")
    except Exception as e:
        raise HTTPException(500, f"Error evaluating model: {str(e)}")



@app.post("/llm")
async def query_llm(request: LLMRequest):
    try:
        async def generate():
            response = client.generate(
                model=request.model,
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return response
        response = await asyncio.create_task(generate())
        return {"response": response["response"]}
    except Exception as e:
        raise HTTPException(500, f"Error querying LLM: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)