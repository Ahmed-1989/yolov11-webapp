
# YOLOv11 Real-Time Object Detection Web App

This project runs a **YOLOv11 object detection model in the browser** using **ONNX.js** and your device's **webcam**, including Safari on iPhone.

## 📦 Files

- `index.html`: Web interface and camera stream setup
- `script.js`: Loads ONNX model, runs real-time detection, and draws bounding boxes
- `yolov11_export.onnx`: Your trained model in ONNX format
- `onnx.min.js`: ONNX runtime for browsers (loaded via CDN)

## 🚀 How to Use

### 1. Upload to GitHub

1. Push all files to a public GitHub repository (e.g., `yolov11-webapp`)
2. Enable **GitHub Pages**:
   - Go to `Settings` → `Pages`
   - Select `main` branch and root folder `/`
   - Save and wait for the deployment URL (e.g., `https://yourusername.github.io/yolov11-webapp/`)

### 2. Run in Browser (iPhone Safari / Desktop Chrome)

1. Open the GitHub Pages link
2. Grant camera access
3. Watch your model detect objects live!

## 🎯 Model Details

- Classes: 15 construction activity/equipment types
- Framework: YOLOv11 (exported from PyTorch to ONNX)
- Inference: ONNX.js
- Preprocessing: [640 x 640], normalized RGB

## 🛠️ Tips

- Detection threshold set to `0.4` in `script.js`
- You can edit `script.js` to tweak box colors, labels, or add NMS
- ONNX model must be small enough to load in mobile browser

## 📃 License

This project is shared for educational and research purposes.
