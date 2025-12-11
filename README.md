# 🥷 Camouflage Object Detection (COD) System
*AI system for detecting objects that blend into their surroundings*

### 🌐 Live Demo  
👉 https://cod-769q.onrender.com/

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)
![Render](https://img.shields.io/badge/Hosted%20On-Render-46e3b7?logo=render)
![License](https://img.shields.io/badge/License-Educational-lightgrey)
![Status](https://img.shields.io/badge/Live-Demo%20Running-brightgreen)

---

## 🎯 Overview
Camouflaged Object Detection (COD) is a challenging computer vision task.  
This project implements **SINet V2** with a **Res2Net-50 backbone** to detect and segment hidden, camouflaged objects.

It features:
- Deep-learning powered inference  
- Web interface with drag-and-drop support  
- Real-time visualization  
- Cloud deployment using Render  

---

## 🧠 How It Works

### 🔍 Architecture
- **Model:** SINet V2  
- **Backbone:** Res2Net-50  
- **Framework:** PyTorch  
- **Server:** FastAPI  
- **Frontend:** HTML + CSS + JavaScript  

### ⚙️ Detection Pipeline
1. User uploads an image  
2. Image is resized → normalized  
3. Multi-scale features extracted via Res2Net  
4. SINet V2 predicts camouflage regions  
5. Outputs generated:
   - Bounding Box View  
   - Segmentation Mask  
   - Heatmap View  

---

## ✨ Key Features
- ⚡ Real-time inference (CPU/GPU)  
- 🔍 Multi-scale detection  
- 📸 Three visualization outputs  
- 📱 Responsive UI  
- 🚀 Render deployment  
- 🎯 Trained on COD10K dataset  

---

## 🏗️ Project Structure (Clean & Correct)

COD/
├── app.py # FastAPI backend server
├── requirements.txt # Python package dependencies
├── runtime.txt # Specifies Python version
├── render.yaml # Render deployment configuration
├── download_models.py # Downloads model weights automatically
│
├── front-end/
│ ├── index.html # Web interface UI
│ ├── style.css # Frontend styling
│ └── script.js # Frontend logic (upload + output)
│
├── back-end/
│ ├── sinetv2_model.py # Model wrapper for inference
│ ├── Network_Res2Net_GRA_NCD.py # SINet V2 architecture (GRA + NCD)
│ └── Res2Net_v1b.py # Res2Net backbone
│
├── models/ # Not stored in Git (auto-downloaded)
│ ├── Net_epoch_best.pth # Main trained model weights
│ └── res2net50_v1b_26w_4s.pth # Backbone weights
│
└── uploads/ # Temporary runtime uploads

yaml
Copy code

✔️ *This structure will render correctly on GitHub.*

---

## 🛠️ Technology Stack

### Backend
- FastAPI  
- PyTorch  
- OpenCV  
- NumPy  
- Pillow  

### Frontend
- HTML5  
- CSS3  
- JavaScript  
- Drag & Drop API  

### Deployment
- Render  
- Git  
- Dropbox (for serving model weights)  

---

## 🚀 Deployment Details (Render)

### Build Command
```bash
pip install -r requirements.txt
Start Command
bash
Copy code
uvicorn app:app --host 0.0.0.0 --port $PORT
Auto Model Download
Runs download_models.py on startup

Downloads SINet V2 + Res2Net weights

Verifies file integrity

🎮 Usage Instructions
🌐 Web Interface
Go to: https://cod-769q.onrender.com

Upload image (Drag-drop / Browse)

Click Analyze Image

View:

Bounding Boxes

Segmentation Mask

Heatmap Output

Download results if needed

🔌 API Endpoints
Endpoint	Method	Description
/	GET	Main UI
/upload	POST	Process image & return results
/health	GET	Status check

🔧 Local Development Setup
Prerequisites
Python 3.11+

(Optional) CUDA GPU

Min. 8 GB RAM

Start Development
bash
Copy code
git clone <repo-url>
cd COD
pip install -r requirements.txt
python app.py
Open browser:
👉 http://localhost:8000

📊 Model Performance
Dataset: COD10K

Input size: 320×320

Avg CPU inference: 2–3 seconds/image

Confidence threshold: 0.01

🔄 Challenges & Solutions
❗ Model File Size Limit
✔️ Solved by hosting on Dropbox + auto download

❗ GPU → CPU Compatibility
✔️ Added map_location='cpu'

❗ PyTorch 2.6 Loading Issues
✔️ Set weights_only=False for compatibility

❗ Google Drive Blocking Downloads
✔️ Moved to stable Dropbox links

🌟 Future Enhancements
GPU inference

Batch processing

Video input support

Mobile app (Android / iOS)

Enhanced visualizations

Custom training UI

🤝 Contributing
Contributions are welcome!
Feel free to open issues or submit feature requests.

📝 License
This project is for educational & research purposes only.
Model architecture credits: SINet V2 & Res2Net papers.

