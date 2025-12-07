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
from typing import List, Optional

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

mobile_model: Optional[models.Model] = None
densenet_model: Optional[models.Model] = None

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

    # Warm-up both models
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


# ------------------ UTILS: LUNG MASK (CLASSIC CV) ------------------


def make_lung_mask(original_rgb: np.ndarray) -> np.ndarray:
    """
    Very simple classical CV lung mask.
    Not perfect but helps suppress ribs / labels / corners.
    Returns mask in [0,1] with same HxW as image.
    """
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    # Improve contrast a bit
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Blur + Otsu threshold
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert: lungs darker → foreground
    thresh = 255 - thresh

    # Morphology to smooth shape
    kernel = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep largest 2 connected components (roughly both lungs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        largest_indices = np.argsort(areas)[-2:] + 1
        lung_mask = np.isin(labels, largest_indices).astype("float32")
    else:
        lung_mask = (closed > 0).astype("float32")

    # Normalize to [0,1]
    lung_mask = cv2.GaussianBlur(lung_mask, (9, 9), 0)
    lung_mask = lung_mask / (lung_mask.max() + 1e-8)
    return lung_mask


# ------------------ UTILS: GENERIC GRAD-CAM ------------------


def _get_last_conv_layer(model: tf.keras.Model):
    """
    Safely find the last convolution-like 4D layer in the whole model.
    Works even if TF renames internal MobileNet/DenseNet layers.
    """
    for layer in reversed(model.layers):
        try:
            shape = layer.output_shape
        except Exception:
            continue
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer
    raise ValueError("❌ No 4D conv layer found for Grad-CAM.")


def make_single_gradcam_heatmap(img_array, model: tf.keras.Model) -> np.ndarray:
    """
    Standard Grad-CAM for a single model.
    img_array: (1,H,W,3) preprocessed
    returns heatmap in [0,1]
    """
    last_conv_layer = _get_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        tape.watch(conv_output)
        top_index = tf.argmax(preds[0])
        top_class = preds[:, top_index]

    grads = tape.gradient(top_class, conv_output)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients are None")

    grads = grads[0]  # (h,w,c)
    conv_output = conv_output[0]  # (h,w,c)

    # Channel-wise importance
    weights = tf.reduce_mean(grads, axis=(0, 1))  # (c,)
    cam = tf.reduce_sum(conv_output * weights, axis=-1)  # (h,w)

    # Normalize to [0,1]
    cam = tf.maximum(cam, 0)
    cam /= tf.reduce_max(cam) + 1e-8
    return cam.numpy()


def make_ensemble_gradcam_heatmap(img_array, models_list: List[tf.keras.Model]) -> np.ndarray:
    """
    If multiple models are provided, compute Grad-CAM for each and average.
    """
    valid_heatmaps = []
    for m in models_list:
        if m is None:
            continue
        try:
            hm = make_single_gradcam_heatmap(img_array, m)
            valid_heatmaps.append(hm)
        except Exception as e:
            print("⚠️ Grad-CAM failed for one model:", e)

    if not valid_heatmaps:
        raise RuntimeError("No valid Grad-CAM heatmaps could be generated.")

    # Resize all to same size & average
    base_h, base_w = valid_heatmaps[0].shape
    acc = np.zeros((base_h, base_w), dtype=np.float32)
    for hm in valid_heatmaps:
        hm_resized = cv2.resize(hm, (base_w, base_h))
        acc += hm_resized.astype("float32")
    acc /= len(valid_heatmaps)

    # Final normalization
    acc = np.maximum(acc, 0)
    acc /= acc.max() + 1e-8
    return acc


def overlay_gradcam(original_rgb: np.ndarray, heatmap: np.ndarray) -> str:
    """
    Overlay heatmap on original RGB image and return base64 PNG string.
    - Applies lung mask to suppress background.
    - Produces a side-by-side (original | overlay) image.

    original_rgb: (H,W,3) uint8
    heatmap: (h,w) [0,1]
    """
    h, w = original_rgb.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply lung mask
    lung_mask = make_lung_mask(original_rgb)  # (H,W) [0,1]
    heatmap_resized *= lung_mask

    # Re-normalize after masking
    if heatmap_resized.max() > 0:
        heatmap_resized /= heatmap_resized.max()

    # To uint8 and apply color map (sharper look)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert original to BGR for OpenCV blending
    orig_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

    # Stronger overlay weights for sharper Grad-CAM
    overlay = cv2.addWeighted(orig_bgr, 0.55, heatmap_color, 0.45, 0)

    # Make side-by-side image: [original | overlay]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w, :] = orig_bgr
    combined[:, w:, :] = overlay

    # Encode as PNG → base64
    success, buffer = cv2.imencode(".png", combined)
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
            models_for_cam = [mobile_model]
            if densenet_model is not None:
                models_for_cam.append(densenet_model)

            heatmap = make_ensemble_gradcam_heatmap(img_array, models_for_cam)
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
