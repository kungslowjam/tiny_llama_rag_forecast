from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
import numpy as np
import os
import torch
from datetime import datetime, timedelta
from ollama import Client
import asyncio

# ตั้งค่าตัวแปรสภาพแวดล้อมสำหรับ Nixtla
os.environ['NIXTLA_ID_AS_COL'] = '1'

# กำหนดโมเดลข้อมูล
class DataPoint(BaseModel):
    ds: str  # เวลา
    y: float  # ค่าจริง
    unique_id: str  # identifier

class RequestData(BaseModel):
    data: List[DataPoint]

class EvaluationMetrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    mape: float
    actual: float
    predicted: float
    timestamp: str

class LLMRequest(BaseModel):
    prompt: str
    model: str = "tinyllama"
    temperature: float = 0.7
    max_tokens: int = 500

# กำหนด FastAPI app
app = FastAPI()

# ��ก็บค่าทำนายล่าสุดไว้ในหน่วยความจำ
last_predictions = {}

# เพิ่มตัวแปร global เก็บโมเดล
global_nn_model = None
last_train_time = None

# กำหนดค่าคงที่
MODEL_DIR = "model_checkpoints"
LATEST_MODEL = "latest_model.pth"
BACKUP_FORMAT = "model_%Y%m%d_%H%M%S.pth"
MAX_BACKUPS = 5

# สร้างโฟลเดอร์เก็บโมเดล
os.makedirs(MODEL_DIR, exist_ok=True)

# สร้าง Ollama client
client = Client(host='http://localhost:11434')

# เพิ่มฟังก์ชันสำหรับโหลดโมเดล
def load_latest_model():
    global global_nn_model
    try:
        latest_model_path = os.path.join(MODEL_DIR, LATEST_MODEL)
        if os.path.exists(latest_model_path):
            print(f"Loading model from {latest_model_path}")
            
            # สร้างโมเดลใหม่
            model = LSTM(
                h=1,
                input_size=24,
                max_steps=100,
                batch_size=32,
                optimizer_kwargs={'lr': 1e-3}
            )
            
            # สร้าง NeuralForecast instance
            global_nn_model = NeuralForecast(
                models=[model],
                freq='5s'
            )
            
            try:
                # โหลด state dict ด้วย weights_only=True
                state_dict = torch.load(latest_model_path, weights_only=True)
                global_nn_model.models[0].load_state_dict(state_dict)
                
                # สร้าง dummy data สำหรับ fit
                dummy_data = pd.DataFrame({
                    'ds': pd.date_range(start='2024-01-01', periods=24, freq='5s'),
                    'y': np.random.rand(24),
                    'unique_id': ['1'] * 24
                })
                
                # ทำ quick fit ด้วย dummy data
                global_nn_model.fit(dummy_data)
                
                print("Model loaded and initialized successfully")
                return True
            except Exception as e:
                print(f"Error initializing model: {str(e)}")
                global_nn_model = None
                return False
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        global_nn_model = None
    return False

# เรียกโหลดโมเดลตอนเริ่มต้น
@app.on_event("startup")
async def startup_event():
    if load_latest_model():
        print("Model loaded on startup")
    else:
        print("No model found or error loading model")

@app.post("/forecast")
async def forecast(request_data: RequestData):
    if not request_data.data:
        raise HTTPException(status_code=422, detail="Invalid input data")
    
    # แปลงข้อมูลเป็น DataFrame
    df = pd.DataFrame([dp.dict() for dp in request_data.data])
    
    # ตรวจสอบข้อมูล
    if df['ds'].isna().any():
        raise HTTPException(status_code=422, detail="Invalid timestamp format")

    # สร้างและ fit โมเดล
    sf = StatsForecast(models=[AutoARIMA()], freq='5s')
    sf.fit(df)
    
    # พยากรณ์
    forecast_df = sf.forecast(df=df, h=1)
    
    # เก็บค่าทำนายและเวลาสุด
    for _, row in forecast_df.iterrows():
        last_predictions[row['unique_id']] = {
            'predicted': row['AutoARIMA'],
            'timestamp': row['ds'],
            'actual': df.iloc[-1]['y']
        }
    
    return forecast_df.to_dict(orient='records')

@app.post("/train_neural")
async def train_neural(request_data: RequestData):
    global global_nn_model, last_train_time
    
    if not request_data.data:
        raise HTTPException(status_code=422, detail="Invalid input data")
    
    try:
        print("Starting neural network training...")
        
        df = pd.DataFrame([dp.dict() for dp in request_data.data])
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        
        print(f"Training data shape: {df.shape}")
        
        if len(df) < 24:
            raise HTTPException(status_code=422, detail=f"Not enough data points. Got {len(df)}, need at least 24")

        # สร้างโมเดล LSTM ใหม่
        model = LSTM(
            h=1,
            input_size=24,
            max_steps=100,
            batch_size=32,
            optimizer_kwargs={'lr': 1e-3}
        )
        
        global_nn_model = NeuralForecast(
            models=[model],
            freq='5s'
        )
        
        # ทำการ fit โมเดล และเก็บค่า loss
        print("Training model...")
        final_loss = global_nn_model.fit(df)
        epoch_loss = float(final_loss) if final_loss is not None else 0.0
        print(f"Final training loss per epoch: {epoch_loss}")
        
        last_train_time = datetime.now()

        # บันทึกโมเดล
        latest_model_path = os.path.join(MODEL_DIR, LATEST_MODEL)
        torch.save({
            'state_dict': global_nn_model.models[0].state_dict(),
            'final_loss': epoch_loss
        }, latest_model_path)

        return {
            "status": "success",
            "trained_at": last_train_time.isoformat(),
            "data_points": len(df),
            "train_loss_epoch": epoch_loss,  # ใช้ค่า loss จริงจากการ train
            "model_path": latest_model_path,
            "message": "Model trained and saved successfully"
        }

    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_neural")
async def predict_neural(request_data: RequestData):
    global global_nn_model, last_train_time, last_predictions
    
    if not request_data.data:
        raise HTTPException(status_code=422, detail="Invalid input data")
        
    try:
        # แปลงข้อมูลเป็น DataFrame
        df = pd.DataFrame([dp.dict() for dp in request_data.data])
        df['ds'] = pd.to_datetime(df['ds'])
        
        print(f"Input data shape: {df.shape}")
        print(f"Sample input:\n{df.head()}")
        
        # ตรวจสอบว่ามีโมเดลหรือไม่
        if global_nn_model is None:
            print("Model not found, attempting to load...")
            if not load_latest_model():
                print("Could not load model, need training first")
                raise HTTPException(status_code=400, detail="Model needs to be trained first")
        
        try:
            # ทำนาย
            forecast = global_nn_model.predict(df)
            print(f"Forecast result:\n{forecast}")
            
            # แปลง timestamp ให้ถูกต้อง
            forecast_time = pd.to_datetime(forecast.index[-1])
            
            # เก็บค่าทำนายล่าสุด
            last_prediction = {
                'predicted': float(forecast.iloc[-1]['LSTM']),
                'ds': forecast_time.strftime('%Y-%m-%d %H:%M:%S'),
                'actual': None
            }
            
            # เก็บใน global context
            last_predictions['1'] = last_prediction
            
            print(f"Stored prediction: {last_prediction}")
            
            return [{
                "unique_id": "1",
                "ds": last_prediction['ds'],
                "LSTM": last_prediction['predicted']
            }]
        except Exception as pred_error:
            print(f"Prediction failed: {str(pred_error)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(pred_error)}")

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(f"Forecast type: {type(forecast) if 'forecast' in locals() else 'Not available'}")
        if 'df' in locals():
            print(f"Input data types:\n{df.dtypes}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate(request_data: RequestData):
    try:
        if not request_data.data:
            raise HTTPException(status_code=422, detail="ไม่มีข้อมูลสำหรับประเมิน")
        
        data_point = request_data.data[0]
        print(f"ข้อมูลที่ได้รับ: {data_point}")
        
        # ตรวจสอบการมีอยู่ของค่าทำนาย
        if not last_predictions or '1' not in last_predictions:
            print("ไม่พบค่าทำนายในระบบ")
            raise HTTPException(status_code=404, detail="ไม่พบค่าทำนายล่าสุด")
        
        pred_data = last_predictions['1']
        print(f"ค่าทำนายที่พบ: {pred_data}")

        # คำนวณค่าต่างๆ
        actual_value = float(data_point.y)
        predicted_value = float(pred_data['predicted'])
        
        mae = abs(actual_value - predicted_value)
        mse = (actual_value - predicted_value) ** 2
        rmse = np.sqrt(mse)
        mape = abs((actual_value - predicted_value) / actual_value) * 100 if actual_value != 0 else 0
        
        return {
            "status": "success",
            "timestamp": data_point.ds,
            "actual": f"{actual_value:.2f}",
            "predicted": f"{predicted_value:.2f}",
            "metrics": {
                "MAE": f"{mae:.2f}",
                "MSE": f"{mse:.2f}",
                "RMSE": f"{rmse:.2f}",
                "MAPE": f"{mape:.2f}"
            }
        }
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาดทั่วไป: {str(e)}")
        return {
            "status": "error",
            "message": f"เกิดข้อผิดพลาด: {str(e)}",
            "request_data": str(request_data)
        }

@app.post("/llm_query")
async def query_llm(request: LLMRequest):
    try:
        # สร้าง coroutine สำหรับเรียก Ollama
        async def generate():
            response = client.generate(
                model=request.model,
                prompt=request.prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return response

        # รัน coroutine
        response = await asyncio.create_task(generate())

        return {
            "response": response['response'],
            "model": request.model,
            "status": "success",
            "total_duration": response.get('total_duration'),
            "load_duration": response.get('load_duration'),
            "prompt_eval_count": response.get('prompt_eval_count'),
            "eval_count": response.get('eval_count')
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying LLM: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
