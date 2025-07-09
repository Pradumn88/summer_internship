from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import traceback
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Added for MobileNetV2 preprocessing

# Import from local module
from model import load_trained_model 

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model when the FastAPI application starts
print("Attempting to load the model for FastAPI...")
try:
    model = load_trained_model()
    # Print model input shape for verification
    print(f"✅ Model loaded successfully. Input shape: {model.input_shape}")
except Exception as e:
    print(f"Failed to load model at FastAPI startup: {e}")
    raise

@app.get("/health")
async def health_check():
    """
    Endpoint to check if the API is running and responsive.
    """
    return {"status": "API is running", "message": "Pneumonia Detection API is live!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to receive an X-ray image, make a pneumonia prediction,
    and return the predicted class and confidence.
    """
    try:
        # 1. Read image contents
        contents = await file.read()

        # 2. Open image using PIL (Pillow)
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 3. Resize image to match model's expected input shape (224x224)
        image = image.resize((224, 224))  # Corrected size

        # 4. Convert PIL Image to NumPy array
        image_array = np.array(image, dtype=np.float32)  # Ensure float32 type

        # 5. Apply MobileNetV2 preprocessing
        image_array = preprocess_input(image_array)

        # 6. Expand dimensions to create a batch of 1 image: (1, 224, 224, 3)
        image_array = np.expand_dims(image_array, axis=0)

        # 7. Make prediction using the loaded model
        prediction = model.predict(image_array)
        class_probabilities = prediction[0]

        # 8. Interpret prediction
        class_labels = ["Normal", "Pneumonia"]
        predicted_class_index = np.argmax(class_probabilities)
        label = class_labels[predicted_class_index]
        confidence = float(class_probabilities[predicted_class_index])

        # Return results
        return {
            "prediction": label,
            "confidence": confidence,
            "probabilities": class_probabilities.tolist()
        }
    
    except Exception as e:
        # Log the full traceback for detailed debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}. Check server logs for more details."
        )