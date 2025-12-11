🥷 Camouflage Object Detection (COD) System
AI System for Detecting Hard-to-See, Camouflaged Objects

🌐 Live Demo: https://cod-769q.onrender.com/

Upload any image and watch the AI uncover hidden, camouflaged objects in real time!

🎯 Overview

Camouflaged Object Detection (COD) is one of the most challenging tasks in computer vision. This project implements a state-of-the-art deep learning pipeline powered by SINet V2 with a Res2Net-50 backbone to accurately detect, segment, and visualize objects that blend seamlessly into their surroundings.

The system is deployed using FastAPI and hosted on Render, offering a responsive web interface where users can upload images and instantly get results.

🧠 How It Works
🔍 Architecture

Model: SINet V2 (Search & Identification Network V2)

Backbone: Res2Net-50 (multi-scale feature extraction)

Framework: PyTorch

Web Server: FastAPI

Frontend: HTML, CSS, JavaScript (Drag & Drop interface)

⚙️ Detection Pipeline

📤 Image Upload — User drags/drops or selects an image

🧽 Preprocessing — Image resized to 320×320, normalized

🧠 Feature Extraction — Res2Net extracts multi-scale patterns

🎯 Detection — SINet V2 finds camouflaged regions

🛠️ Post-processing — Produces masks, bounding boxes, heatmaps

👁️ Visualization — Generates three outputs:

Bounding Box View

Segmentation Mask View

Heatmap Probability View

✨ Key Features

⚡ Real-time detection on Render (CPU/GPU compatible)

🔭 Multi-scale analysis for detecting large & small camouflaged objects

🎯 High Accuracy trained on the COD10K dataset

🖼️ Interactive Web UI with drag-and-drop upload

🎨 Multiple visualization modes

📱 Responsive design (works on mobile & desktop)

🏗️ Project Structure
COD/
├── app.py                          # FastAPI backend server
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version
├── render.yaml                     # Render deployment config
├── download_models.py              # Downloads model weights
│
├── Front End/
│   ├── index.html                  # UI page
│   ├── style.css                   # Styles
│   └── script.js                   # JS functionality
│
├── Back End/
│   ├── sinetv2_model.py            # Model wrapper
│   ├── Network_Res2Net_GRA_NCD.py  # Architecture implementation
│   └── Res2Net_v1b.py              # Res2Net backbone
│
├── COD10K Trained model/
│   ├── Net_epoch_best.pth          # Trained COD model
│   └── res2net50_v1b_26w.pth       # Backbone weights
│
└── uploads/                        # Temporary uploaded images

🛠️ Technology Stack
Backend

⚡ FastAPI

🔥 PyTorch

🖼️ OpenCV

🧮 NumPy

🖌️ Pillow

Frontend

HTML5

CSS3

JavaScript

Drag & Drop API

Deployment

☁️ Render (cloud hosting)

Git (version control)

Dropbox (model hosting)

🚀 Deployment Workflow
1️⃣ Model Preparation

Trained on COD10K

Model weights stored on Dropbox

Auto-download during first server startup

2️⃣ Code Optimization

GPU → CPU conversion for Render

Added stable error handling

Added fallback loading mechanisms

3️⃣ Render Deployment

Build Command:
pip install -r requirements.txt

Start Command:
uvicorn app:app --host 0.0.0.0 --port $PORT

Python version: 3.11.9

4️⃣ Automatic Weight Download

SINet V2 backbone downloaded from official source

Trained weights downloaded from Dropbox

Validation checks ensure correct weights

🎮 Usage Instructions
🌐 Web Interface

Visit → https://cod-769q.onrender.com

Upload an image (drag/drop or browse)

Click Analyze Image

View results:

Detection

Segmentation

Heatmap

Download images if needed

🧪 API Endpoints
Endpoint	Method	Description
/	GET	Main UI
/upload	POST	Image detection
/health	GET	System health check
/style.css	GET	Stylesheet
/script.js	GET	Frontend JavaScript
🔧 Local Development
Prerequisites

Python 3.11+

(Optional) CUDA-based GPU

8GB RAM recommended

Setup
git clone <repository_url>
pip install -r requirements.txt
python app.py


Open browser:
👉 http://localhost:8000

📊 Model Performance

Dataset: COD10K

Architecture: SINet V2 + Res2Net-50

Input Size: 320 × 320

Avg Inference Time: 2–3 seconds (CPU)

Confidence Threshold: 0.01

🌟 Key Innovations

🎯 Group-Reversal Attention (GRA)

🔄 Neighbor Connection Decoder (NCD)

📡 Multi-Scale Feature Extraction

⚡ Fast real-time web inference

🔄 Deployment Challenges & Solutions
❗ Large model files

✔️ Used cloud download instead of storing in Git

❗ GPU → CPU migration

✔️ Added map_location='cpu' and correct PyTorch flags

❗ PyTorch 2.6 loading issues

✔️ Used weights_only=False for compatibility

❗ Google Drive blocked downloads

✔️ Switched to stable Dropbox direct links

🎯 Future Enhancements

🚀 GPU inference support

🎥 Video camouflaged object detection

📦 Batch image processing

📱 Android/iOS app

🎨 Advanced visualization modes

🧰 Custom training UI

📝 License

This project is intended for educational and research purposes.
Model weights and architecture follow their respective research publications.

🤝 Contributing

Contributions are always welcome!
Feel free to open an issue or submit a pull request ⭐
