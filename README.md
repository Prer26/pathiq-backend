# PathIQ Road Perception API

## Local setup (Windows PowerShell)

```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Health check: `http://127.0.0.1:8000/health`
API docs: `http://127.0.0.1:8000/docs`

Configure the frontend with `VITE_PATHIQ_API_URL=http://127.0.0.1:8000`. Restart Vite after changing `.env`.

## Included model weights

This project bundle includes `models/best_model3 (2).pth` for U-Net road segmentation and `models/yolo_detector.pt` for YOLO detection. The API loads these paths by default. Keep large model files out of Git; use a model artifact or mounted file for deployment.

## Deployment notes

Deploy this folder as a Python web service and set `SEGMENTATION_MODEL_PATH` and (optionally) `YOLO_MODEL_PATH` to accessible weight-file paths. For a public service, set `CORS_ORIGINS` to the exact frontend origin. Large model weights should be stored as a deployment artifact or mounted persistent file, not committed to Git. A CPU-only free tier may be too slow for live inference; GPU hosting or client-side inference may be needed for practical frame rates.

The browser sends camera frames to this API for inference. Inform users and obtain consent before enabling this upload. Do not use this prototype as the sole basis for vehicle control or safety decisions.
