# FastAPI Forecasting and LLM API for Raspberry Pi 5

This FastAPI application provides forecasting services using statistical and neural network models, as well as integration with an LLM API (Ollama). It includes endpoints for data forecasting, training neural models, performing predictions, evaluating results, and querying an LLM.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration for Raspberry Pi 5](#configuration-for-raspberry-pi-5)
- [Endpoints](#endpoints)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Model Persistence](#model-persistence)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [License](#license)

---

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

---

## Requirements
### Hardware
- Raspberry Pi 5 (ARM architecture with 8GB RAM recommended)

### Software
- **Python 3.10+** (Raspberry Pi OS)
- Libraries:
  - FastAPI
  - Uvicorn
  - pandas
  - statsforecast
  - neuralforecast
  - torch (compatible with ARM, using PyTorch Lite)
  - Ollama client
  - NumPy

Install dependencies:
```bash
pip install fastapi uvicorn pandas statsforecast neuralforecast ollama numpy
```

**ARM-Compatible Torch Installation**:
On ARM devices like Raspberry Pi, install PyTorch via:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Configuration for Raspberry Pi 5

### 1. Optimizing Tensor Operations
Enable PyTorch to use optimized tensor operations for ARM CPUs:
```python
torch.set_num_threads(4)  # Adjust based on CPU cores
```

### 2. Lightweight Models
- Use smaller neural network models and adjust hyperparameters for limited hardware.
- Reduce batch sizes and sequence lengths for LSTM models during training:
  ```python
  batch_size = 8
  seq_length = 24  # Default for time series data
  ```

### 3. Ollama LLM
Ensure Ollama server is running on the Raspberry Pi:
1. Install Ollama server for ARM-compatible Linux from their [official page](https://ollama.ai/download).
2. Start the server:
   ```bash
   ollama serve --model tinyllama
   ```
3. Verify it's running at `http://localhost:11434`.

---

## Endpoints
### 1. Forecast Time Series Data
**POST /forecast**
- Input: List of data points with `ds` (timestamp), `y` (value), and `unique_id`.
- Output: Forecasted values using AutoARIMA.

...

*(End-to-end API details remain unchanged. Refer to the original document.)*

---

## Running the Application
Run the FastAPI server on the Raspberry Pi:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Documentation
- Swagger UI: [http://<raspberrypi_ip>:8000/docs](http://<raspberrypi_ip>:8000/docs)
- ReDoc: [http://<raspberrypi_ip>:8000/redoc](http://<raspberrypi_ip>:8000/redoc)

---

## Testing
Use tools like **Postman**, **cURL**, or local scripts to test the API on the Raspberry Pi.

Example with `curl`:
```bash
curl -X POST "http://<raspberrypi_ip>:8000/forecast" \
-H "Content-Type: application/json" \
-d '{ "data": [{"ds": "2024-06-01T00:00:00", "y": 10.5, "unique_id": "1"}] }'
```

---

## License
This project is licensed under the MIT License.
