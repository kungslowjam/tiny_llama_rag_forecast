# FastAPI Forecasting and LLM API

This FastAPI application provides forecasting services using statistical and neural network models, as well as integration with an LLM API (Ollama). It includes endpoints for data forecasting, training neural models, performing predictions, evaluating results, and querying an LLM.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Endpoints](#endpoints)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Model Persistence](#model-persistence)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [License](#license)

## Features
1. **Time Series Forecasting**: 
   - Uses `AutoARIMA` for statistical forecasting.
   - Implements LSTM neural networks for deep learning-based forecasting.
2. **Model Training and Persistence**:
   - Neural models are trained, saved, and reloaded from checkpoints.
3. **Evaluation Metrics**:
   - MAE, MSE, RMSE, and MAPE for accuracy evaluation.
4. **Large Language Model Query**:
   - Queries an Ollama LLM (default: `tinyllama`) for generating text responses.
5. **Error Handling**:
   - Robust validation and error responses for all endpoints.

## Requirements
- **Python 3.8+**
- FastAPI
- Uvicorn
- pandas
- statsforecast
- neuralforecast
- torch
- Ollama client
- NumPy

Install dependencies using `pip`:
```bash
pip install fastapi uvicorn pandas statsforecast neuralforecast torch ollama numpy
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is running on `http://localhost:11434` for LLM queries.

## Endpoints
### 1. Forecast Time Series Data
**POST /forecast**
- Input: List of data points with `ds` (timestamp), `y` (value), and `unique_id`.
- Output: Forecasted values using AutoARIMA.

**Request**:
```json
{
  "data": [
    { "ds": "2024-06-01T00:00:00", "y": 10.5, "unique_id": "1" }
  ]
}
```
**Response**:
```json
[
  { "unique_id": "1", "AutoARIMA": 12.4, "ds": "2024-06-01T00:00:05" }
]
```

### 2. Train Neural Network (LSTM)
**POST /train_neural**
- Input: Training data (minimum 24 points).
- Output: Training loss, training timestamp, and saved model path.

**Request**:
```json
{
  "data": [
    { "ds": "2024-06-01T00:00:00", "y": 10.5, "unique_id": "1" },
    ...
  ]
}
```
**Response**:
```json
{
  "status": "success",
  "trained_at": "2024-06-01T12:00:00",
  "train_loss_epoch": 0.005,
  "model_path": "model_checkpoints/latest_model.pth"
}
```

### 3. Predict Using Neural Network (LSTM)
**POST /predict_neural**
- Input: Time series data for prediction.
- Output: Predicted value with a timestamp.

**Request**:
```json
{
  "data": [
    { "ds": "2024-06-01T00:00:00", "y": 10.5, "unique_id": "1" }
  ]
}
```
**Response**:
```json
[
  { "unique_id": "1", "ds": "2024-06-01T00:00:05", "LSTM": 11.8 }
]
```

### 4. Evaluate Prediction
**POST /evaluate**
- Input: Actual data point for comparison.
- Output: Evaluation metrics (MAE, MSE, RMSE, MAPE).

**Request**:
```json
{
  "data": [
    { "ds": "2024-06-01T00:00:05", "y": 12.0, "unique_id": "1" }
  ]
}
```
**Response**:
```json
{
  "status": "success",
  "actual": "12.00",
  "predicted": "11.80",
  "metrics": {
    "MAE": "0.20",
    "MSE": "0.04",
    "RMSE": "0.20",
    "MAPE": "1.67"
  }
}
```

### 5. Query LLM
**POST /llm_query**
- Input: Prompt, model name, temperature, and token limit.
- Output: Generated LLM response.

**Request**:
```json
{
  "prompt": "Explain time series forecasting.",
  "model": "tinyllama",
  "temperature": 0.7,
  "max_tokens": 200
}
```
**Response**:
```json
{
  "response": "Time series forecasting is ...",
  "model": "tinyllama",
  "status": "success"
}
```

## Environment Variables
- **NIXTLA_ID_AS_COL**: Set to `1` to define column settings for Nixtla models.

## Model Persistence
- Models are saved to `model_checkpoints/` directory.
- **latest_model.pth**: Default saved model file.
- Automatic backups ensure older versions are saved based on timestamps.

## Usage
1. Send input data to `/forecast` for statistical predictions.
2. Train a neural model using `/train_neural`.
3. Predict using `/predict_neural`.
4. Evaluate predictions with `/evaluate`.
5. Query the LLM with `/llm_query`.

## Running the Application
Run the FastAPI server locally:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access the API docs:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Testing
Use tools like **Postman** or **cURL** to test the endpoints.

Example with `curl`:
```bash
curl -X POST "http://localhost:8000/forecast" -H "Content-Type: application/json" -d '{ "data": [{"ds": "2024-06-01T00:00:00", "y": 10.5, "unique_id": "1"}] }'
```

## License
This project is licensed under the MIT License.
