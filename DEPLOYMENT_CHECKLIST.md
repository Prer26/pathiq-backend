# PathIQ perception deployment checklist

## Required artifacts
- `best_model3 (2).pth` — trained U-Net/ResNet-50 road-segmentation weights.
- `yolo_detector.pt` — optional YOLO weights for object detection boxes.

Place these in `backend/models/` for local development, or mount/store them on the backend host and set `SEGMENTATION_MODEL_PATH` / `YOLO_MODEL_PATH` accordingly. Do not commit large model weights to a public repository.

## Local run (Windows PowerShell)
```powershell
cd backend
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

In the project root, set `VITE_PATHIQ_API_URL=http://127.0.0.1:8000` in `.env.local`, then run:
```powershell
npm ci
npm run dev
```

Open `/health` first. `segmentation_loaded` must be `true` before the frontend can display real AI output. `yolo_loaded` will be `true` only when YOLO dependencies and weights are available.

## Frontend configuration for deployment
Set `VITE_PATHIQ_API_URL` to the deployed backend's HTTPS base URL in the frontend hosting environment and rebuild the frontend. Set backend `CORS_ORIGINS` to the exact deployed frontend origin(s), comma-separated.

The browser camera requires HTTPS in production (except localhost). The browser sends sampled camera frames to the backend for inference. Display a clear notice and obtain user consent before enabling camera uploads.

## Limitations
- The model weights are not included in this ZIP.
- A CPU-only host may deliver low inference FPS; test with the real weights before promising real-time performance.
- This prototype is for perception visualization only, not autonomous vehicle control or a safety-critical driver-assistance system.
