from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import logging

# Inisialisasi FastAPI
app = FastAPI(title="Tanaman Padi Prediction API (XGBoost)")

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load model XGBoost dari file pickle
try:
    with open("XGBoost_Regressor_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    logging.info("✅ Model XGBoost berhasil dimuat.")
except Exception as e:
    logging.error(f"❌ Gagal memuat model: {e}")
    xgb = None

# Load MinMaxScaler model dari pickle
try:
    with open("minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logging.info("✅ Model MinMaxScaler berhasil dimuat.")
except Exception as e:
    logging.error(f"❌ Gagal memuat model MinMaxScaler: {e}")
    scaler = None

# Input schema sesuai nama kolom saat training
class CropData(BaseModel):
    Provinsi: int
    Tahun: int
    Produksi: float
    Luas_Panen: float
    Curah_hujan: float
    Kelembapan: float
    Suhu_rata_rata: float

# Output schema untuk hasil prediksi
class PredictionResponse(BaseModel):
    prediction: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "✅ XGBoost API for Tanaman Padi Prediction is running"}

# Preprocessing input
def preprocess_input(data: CropData):
    # Convert input ke DataFrame
    df = pd.DataFrame([data.dict()])
    logging.info(f"Input data: {df}")

    # Pemetaan nama kolom agar sesuai dengan nama kolom saat model dilatih
    column_mapping = {
        "Luas_Panen": "Luas Panen",
        "Curah_hujan": "Curah hujan",
        "Suhu_rata_rata": "Suhu rata-rata"
    }
    df.rename(columns=column_mapping, inplace=True)

    # Periksa apakah ada kolom yang hilang atau tidak sesuai dengan model
    expected_columns = ['Provinsi', 'Tahun', 'Produksi', 'Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {', '.join(missing_cols)}")

    if scaler is None:
        raise ValueError("Scaler belum dimuat. Pastikan file 'minmax_scaler.pkl' tersedia.")
    
    # Apply MinMax normalization (using pre-loaded scaler)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    logging.info(f"Normalized input data: {df_scaled}")
    
    return df_scaled

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_crop_yield(data: CropData):
    try:
        if xgb is None:
            raise ValueError("Model belum dimuat. Pastikan file 'xgboost_model.pkl' tersedia.")
        
        # Preprocess and normalize input data
        processed = preprocess_input(data)
        
        # Predict using the XGBoost model
        prediction = xgb.predict(processed)[0]
        
        return {"prediction": round(float(prediction), 2)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}
