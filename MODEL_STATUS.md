# Model file status

- `models/best_model3 (2).pth`: included. Inspected as a PyTorch state dictionary with 380 parameter/buffer entries, including ResNet encoder keys.
- `models/yolo_detector.pt`: included. File container structure is readable as a PyTorch checkpoint archive.
- Runtime inference has **not** been verified in this environment because `segmentation-models-pytorch` and `ultralytics` are not installed here. Start the backend and check `/health` before relying on inference.
- The frontend sends camera frames to the backend. Use HTTPS in production and configure `CORS_ORIGINS` to the exact deployed frontend origin.
