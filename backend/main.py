from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import models
from PIL import Image
import io
import csv
import base64
import cv2
from datetime import datetime

import os

print("FILES IN /app/models:", os.listdir("models"))
print("CURRENT DIRECTORY:", os.getcwd())


# ------------------ PATHS & CONSTANTS ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MOBILE_MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_xray_model.keras")
DENSENET_MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_xray_densenet.keras")  # optional
LOG_FILE = os.path.join(LOG_DIR, "predictions.csv")

CLASS_NAMES = ["COVID", "NORMAL", "PNEUMONIA", "TB"]
IMG_SIZE = (224, 224)

mobile_model = None
densenet_model = None

# ------------------ FASTAPI SETUP ------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ UTILS: MODEL LOADING ------------------


def load_models():
    """Load MobileNetV2 (and optional DenseNet) into memory and warm them up."""
    global mobile_model, densenet_model

    if not os.path.exists(MOBILE_MODEL_PATH):
        raise RuntimeError(
            f"MobileNet model not found at {MOBILE_MODEL_PATH}. "
            "Run model.py training first."
        )

    mobile_model = tf.keras.models.load_model(MOBILE_MODEL_PATH, compile=False)
    print("✅ Loaded MobileNetV2 model from", MOBILE_MODEL_PATH)

    if os.path.exists(DENSENET_MODEL_PATH):
        densenet_model = tf.keras.models.load_model(
            DENSENET_MODEL_PATH, compile=False
        )
        print("✅ Loaded DenseNet ensemble model from", DENSENET_MODEL_PATH)
    else:
        densenet_model = None
        print("ℹ️ DenseNet model not found, using only MobileNetV2.")

    # Warm-up
    dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    mobile_model.predict(dummy)
    if densenet_model is not None:
        densenet_model.predict(dummy)

    print("✅ Models warmed up. Classes:", CLASS_NAMES)


# ------------------ UTILS: IMAGE PREP ------------------


def preprocess_image(image_bytes):
    """Load uploaded bytes → RGB, resize, return (model_input, original_rgb)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    original_rgb = np.array(img_resized)

    x = original_rgb.astype("float32")
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    return x, original_rgb


# ------------------ UTILS: GENERIC GRAD-CAM ------------------


def _get_last_conv_layer(model):
    """
    Automatically find the last conv-like layer with 4D output.
    Works even if layer names change.
    """
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
        except Exception:
            continue
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer
    raise ValueError("No 4D convolutional layer found for Grad-CAM.")


def make_gradcam_heatmap(img_array, model):
    """
    Generic Grad-CAM implementation.

    img_array: (1, H, W, 3) preprocessed input
    model: keras.Model
    """
    last_conv_layer = _get_last_conv_layer(model)

    # Model that maps input -> (last conv feature maps, predictions)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)     # conv_output: (1, h, w, c)
        tape.watch(conv_output)                        # IMPORTANT
        top_index = tf.argmax(preds[0])
        top_channel = preds[:, top_index]

    # Gradients of top predicted class wrt conv_output
    grads = tape.gradient(top_channel, conv_output)    # (1, h, w, c)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients are None")

    grads = grads[0]                                   # (h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))  # (c,)

    conv_output = conv_output[0]                       # (h, w, c)
    conv_output = conv_output * pooled_grads

    heatmap = tf.reduce_mean(conv_output, axis=-1)     # (h, w)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()



def overlay_gradcam(original_rgb, heatmap):
    """
    Overlay heatmap on original RGB image and return base64 PNG string.
    original_rgb: (H, W, 3), uint8 or float
    heatmap: (h, w) [0,1]
    """
    h, w = original_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    orig_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

    success, buffer = cv2.imencode(".png", overlay)
    if not success:
        raise RuntimeError("Failed to encode Grad-CAM image.")
    return base64.b64encode(buffer).decode("utf-8")


# ------------------ UTILS: LOGGING ------------------


def log_prediction(filename, label, confidence):
    """Append prediction metadata to logs/predictions.csv."""
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "file", "label", "confidence"])
        writer.writerow(
            [datetime.now().isoformat(), filename, label, f"{confidence:.4f}"]
        )


# ------------------ FASTAPI EVENTS ------------------


@app.on_event("startup")
async def on_startup():
    try:
        load_models()
    except Exception as e:
        print("❌ Error loading models on startup:", e)


# ------------------ ROUTES ------------------


@app.get("/")
async def root():
    models_used = ["MobileNetV2"]
    if densenet_model is not None:
        models_used.append("DenseNet121 (ensemble)")
    return {
        "message": "Chest X-Ray Multi-disease API with Grad-CAM & Ensemble",
        "model_loaded": mobile_model is not None,
        "classes": CLASS_NAMES,
        "models": models_used,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if mobile_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please start server after training.",
        )

    try:
        image_bytes = await file.read()
        img_array, original_rgb = preprocess_image(image_bytes)

        # --- predictions (MobileNet + optional DenseNet ensemble) ---
        preds_mobile = mobile_model.predict(img_array)
        if densenet_model is not None:
            preds_dense = densenet_model.predict(img_array)
            preds = (preds_mobile + preds_dense) / 2.0
        else:
            preds = preds_mobile

        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = CLASS_NAMES[idx]
        probabilities = preds[0].tolist()  # list[float]

        # --- Grad-CAM (non-fatal) ---
        gradcam_b64 = None
        try:
            heatmap = make_gradcam_heatmap(img_array, mobile_model)
            gradcam_b64 = overlay_gradcam(original_rgb, heatmap)
        except Exception as e:
            # We still return prediction even if Grad-CAM fails
            print("⚠️ Grad-CAM generation error:", e)

        # --- Log prediction ---
        log_prediction(file.filename, label, confidence)

        return {
            "prediction": label,
            "confidence": confidence,   # 0–1, frontend multiplies by 100
            "probabilities": probabilities,
            "gradcam": gradcam_b64,    # base64 string or null
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Prediction error:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}",
        )
