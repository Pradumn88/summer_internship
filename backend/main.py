from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import traceback
import os
import platform
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Import from local module
from model import load_trained_model, predict_image

app = FastAPI()

# Enable CORS for frontend communication
# In backend/main.py

app.add_middleware(
    CORSMiddleware,
    # Add your Vercel URL to this list
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://pneumonia-prediction-amber.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_labels = ["NORMAL", "PNEUMONIA"]

# --------------------------------------------------
# MAC-SPECIFIC OPTIMIZATIONS
# --------------------------------------------------
if platform.system() == 'Darwin':
    # Configure for Apple Silicon
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_METAL_ENABLED'] = '1'
    print("🍎 Configured for Apple Silicon")

@app.on_event("startup")
async def startup_event():
    """Load model and configure for macOS"""
    global model
    
    print("🔄 Starting FastAPI... loading model...")
    
    try:
        # Try to load trained model
        model = load_trained_model()
        print(f"✅ Model loaded successfully")
        
        # Warm up model
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = model.predict(dummy_input)
        print("🔥 Model warmed up")
        
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("❗ Starting without model - training required")
        model = None

@app.get("/")
async def read_root():
    """Root endpoint with system info"""
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "processor": platform.processor(),
        "tensorflow_version": tf.__version__,
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
    }
    return {
        "message": "Pneumonia Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "system_info": system_info
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
        "system": platform.platform()
    }

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Enhanced prediction endpoint with uncertainty"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train or provide a model file."
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save temp file for model prediction
        temp_path = f"/tmp/{file.filename}"
        image.save(temp_path)
        
        # Predict with uncertainty estimation
        prediction, confidence, uncertainty, probabilities = predict_image(
            model, temp_path, class_labels, num_samples=5
        )
        
        # Clean up
        os.remove(temp_path)
        
        if prediction is None:
            raise HTTPException(status_code=400, detail="Prediction failed")
        
        # Return confidence as raw probability (0-1)
        return {
            "prediction": prediction,
            "confidence": confidence / 100,  # Convert back to 0-1 range
            "uncertainty": uncertainty,
            "probabilities": probabilities
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )