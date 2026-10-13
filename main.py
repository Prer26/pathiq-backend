
import os
import time
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pathiq")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

SEGMENTATION_MODEL_PATH = MODEL_DIR / "best_model3 (2).pth"
YOLO_MODEL_PATH = MODEL_DIR / "yolo_detector.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_ALPHA = 0.40
ROAD_THRESHOLD = 0.50
MAX_UPLOAD_SIZE = 15 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1920

segmentation_model = None
yolo_model = None

segmentation_error = None
yolo_error = None

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_segmentation_model():
    global segmentation_model, segmentation_error

    try:
        if not SEGMENTATION_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Segmentation model not found: {SEGMENTATION_MODEL_PATH}"
            )

        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        checkpoint = torch.load(
            SEGMENTATION_MODEL_PATH,
            map_location=DEVICE,
        )

        # Support common checkpoint formats.
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

        if not isinstance(checkpoint, dict):
            raise ValueError("Unexpected segmentation checkpoint format.")

        # Remove common DataParallel prefixes.
        cleaned_checkpoint = {}
        for key, value in checkpoint.items():
            new_key = key.removeprefix("module.")
            cleaned_checkpoint[new_key] = value

        model.load_state_dict(cleaned_checkpoint)
        model.to(DEVICE)
        model.eval()

        segmentation_model = model
        segmentation_error = None

        logger.info("Segmentation model loaded successfully.")

    except Exception as exc:
        segmentation_model = None
        segmentation_error = str(exc)
        logger.exception("Failed to load segmentation model.")


def load_yolo_model():
    global yolo_model, yolo_error

    try:
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {YOLO_MODEL_PATH}"
            )

        yolo_model = YOLO(str(YOLO_MODEL_PATH))
        yolo_error = None

        logger.info("YOLO model loaded successfully.")

    except Exception as exc:
        yolo_model = None
        yolo_error = str(exc)
        logger.exception("Failed to load YOLO model.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PathIQ on %s", DEVICE)
    load_segmentation_model()
    load_yolo_model()
    yield
    logger.info("PathIQ shutting down.")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="PathIQ Road Perception API",
    description="Road segmentation and object detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# Localhost development origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------

def decode_image(image_bytes: bytes):
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Invalid image. Upload a valid JPG or PNG.")

    height, width = frame.shape[:2]

    if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        scale = min(
            MAX_IMAGE_DIMENSION / width,
            MAX_IMAGE_DIMENSION / height,
        )

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def run_segmentation(frame):
    if segmentation_model is None:
        raise RuntimeError(
            segmentation_error or "Segmentation model is unavailable."
        )

    # OpenCV provides BGR; convert to RGB for the model.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to a practical inference resolution.
    input_size = (512, 512)
    resized = cv2.resize(
        rgb,
        input_size,
        interpolation=cv2.INTER_LINEAR,
    )

    # Assumes model training used RGB images normalized to [0, 1].
    tensor = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = segmentation_model(tensor)
        probabilities = torch.sigmoid(logits)[0, 0]
        probabilities = probabilities.cpu().numpy()

    original_height, original_width = frame.shape[:2]

    probabilities = cv2.resize(
        probabilities,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    mask = (probabilities >= ROAD_THRESHOLD).astype(np.uint8)

    return mask


def run_yolo(frame):
    if yolo_model is None:
        return [], yolo_error

    detections = []

    try:
        results = yolo_model.predict(
            source=frame,
            conf=0.25,
            verbose=False,
        )

        for result in results:
            if result.boxes is None:
                continue

            names = result.names

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                confidence = float(box.conf[0].cpu())
                class_id = int(box.cls[0].cpu())

                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                else:
                    class_name = (
                        names[class_id]
                        if class_id < len(names)
                        else str(class_id)
                    )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                })

    except Exception:
        logger.exception("YOLO inference failed.")
        return [], "YOLO inference failed."

    return detections, None


def create_visualization(frame, mask, detections):
    output = frame.copy()

    # Green road segmentation overlay.
    green_layer = np.zeros_like(output)
    green_layer[:, :] = (0, 255, 0)

    road_pixels = mask.astype(bool)

    if np.any(road_pixels):
        blended = cv2.addWeighted(
            output,
            1.0 - MASK_ALPHA,
            green_layer,
            MASK_ALPHA,
            0,
        )
        output[road_pixels] = blended[road_pixels]

    # Yellow contours around the road mask.
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 255),
        2,
    )

    # Red YOLO bounding boxes.
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )

        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )

        label_y = max(20, y1 - 8)

        cv2.putText(
            output,
            label,
            (max(0, x1), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def encode_image(frame):
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )

    if not success:
        raise RuntimeError("Could not encode processed image.")

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "PathIQ Road Perception API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "PathIQ Road Perception API",
        "device": str(DEVICE),
        "segmentation_loaded": segmentation_model is not None,
        "segmentation_error": segmentation_error,
        "yolo_loaded": yolo_model is not None,
        "yolo_error": yolo_error,
    }


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)):
    started_at = time.perf_counter()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read(MAX_UPLOAD_SIZE + 1)

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 15 MB upload limit.",
        )

    try:
        frame = decode_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if segmentation_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Segmentation model is unavailable.",
                "error": segmentation_error,
            },
        )

    try:
        mask = run_segmentation(frame)
        detections, detection_error = run_yolo(frame)

        visualization = create_visualization(
            frame,
            mask,
            detections,
        )

        processed_image = encode_image(visualization)

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        road_percentage = float(np.mean(mask) * 100)

        return {
            "success": True,
            "image": processed_image,
            "image_base64": processed_image,
            "road_percentage": round(road_percentage, 2),
            "detections": detections,
            "detection_count": len(detections),
            "inference_ms": round(elapsed_ms, 2),
            "fps": round(1000 / elapsed_ms, 2) if elapsed_ms > 0 else 0,
            "device": str(DEVICE),
            "yolo_enabled": yolo_model is not None,
            "yolo_error": detection_error,
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Frame processing failed.")
        raise HTTPException(
            status_code=500,
            detail="Frame processing failed. Check the backend terminal.",
        )
    finally:
        await file.close()


# --------------------------------------------------
# LOCAL DEVELOPMENT
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
