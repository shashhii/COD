# Camouflage Object Detection (COD) System

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Deployment](https://img.shields.io/badge/deployed-Render-purple.svg)

## 🎯 Overview

This project implements a state-of-the-art **Camouflage Object Detection** system using deep learning to identify objects that blend seamlessly with their surroundings. The system uses the **SINet V2** architecture with **Res2Net** backbone for accurate detection and segmentation of camouflaged objects.

## 🚀 Live Demo

**Deployed Application**: [https://cod-769q.onrender.com](https://cod-769q.onrender.com)

Upload any image and watch the AI detect camouflaged objects in real-time!

## 🛠️ Local Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Git
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, CPU works too)

### Step 1: Clone Repository
```bash
git clone https://github.com/shashhii/COD.git
cd COD
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Weights
The application will automatically download required model files on first run:
- **Res2Net backbone**: ~100MB
- **Trained COD model**: ~400MB

### Step 5: Run Application
```bash
python app.py
```

### Step 6: Access Application
Open your browser and navigate to:
- **Local URL**: `http://localhost:8000`
- **Alternative**: `http://127.0.0.1:8000`

### Troubleshooting Local Setup

**Port Already in Use:**
```bash
# Use different port
set PORT=8001 && python app.py  # Windows
export PORT=8001 && python app.py  # macOS/Linux
```

**Model Download Issues:**
- Ensure stable internet connection
- Models download automatically on first startup
- Check `COD10K Trained model/` directory for downloaded files

**Memory Issues:**
- Close other applications
- Use CPU-only mode (default)
- Reduce image size if processing fails

## 🧠 How It Works

### Architecture
- **Model**: SINet V2 (Search & Identification Network V2)
- **Backbone**: Res2Net-50 with multi-scale feature extraction
- **Framework**: PyTorch for deep learning, FastAPI for web service
- **Frontend**: HTML/CSS/JavaScript with drag-and-drop interface

### Detection Process
1. **Image Upload**: User uploads an image through the web interface
2. **Preprocessing**: Image is resized to 320x320 and normalized
3. **Feature Extraction**: Res2Net backbone extracts multi-scale features
4. **Detection**: SINet V2 identifies camouflaged regions
5. **Post-processing**: Generates bounding boxes, masks, and confidence scores
6. **Visualization**: Creates three types of output:
   - **Detection View**: Bounding boxes with confidence scores
   - **Segmentation View**: Pixel-perfect masks overlay
   - **Heatmap View**: Probability distribution visualization

### Key Features
- **Real-time Detection**: Fast inference on CPU/GPU
- **Multi-scale Analysis**: Detects objects of various sizes
- **High Accuracy**: Trained on COD10K dataset
- **Interactive Interface**: Drag-and-drop image upload
- **Multiple Visualizations**: Bounding boxes, segmentation masks, heatmaps
- **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Project Structure

```
COD/
├── app.py                          # Main FastAPI application
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version specification
├── render.yaml                     # Render deployment configuration
├── download_models.py              # Model download script
├── Front End/                      # Web interface
│   ├── index.html                  # Main HTML page
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend JavaScript
├── Back End/                       # AI model implementation
│   ├── sinetv2_model.py           # SINet V2 model wrapper
│   ├── Network_Res2Net_GRA_NCD.py # Network architecture
│   └── Res2Net_v1b.py             # Res2Net backbone
├── COD10K Trained model/          # Pre-trained model weights (auto-downloaded)
│   ├── Net_epoch_best.pth         # Main trained model
│   └── res2net50_v1b_26w_4s-3cf99910.pth # Backbone weights
└── uploads/                        # Temporary upload directory
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance web framework
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations
- **Pillow**: Image processing

### Frontend
- **HTML5**: Structure and layout
- **CSS3**: Styling and animations
- **JavaScript**: Interactive functionality
- **Drag & Drop API**: File upload interface

### Deployment
- **Render**: Cloud hosting platform
- **Git**: Version control
- **Dropbox**: Model file hosting

## 🚀 Deployment Process

### 1. Model Preparation
- Trained SINet V2 model on COD10K dataset
- Uploaded model weights to Dropbox for reliable downloading
- Configured automatic model download during deployment

### 2. Code Optimization
- Converted from GPU to CPU-only PyTorch for cloud deployment
- Added error handling for model loading failures
- Implemented fallback mechanisms for robust operation

### 3. Render Deployment
- **Platform**: Render.com (free tier)
- **Runtime**: Python 3.11.9
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 4. Automatic Model Download
- Models download automatically on first startup
- Res2Net backbone: Downloaded from official source
- Trained weights: Downloaded from Dropbox
- Validation checks ensure model integrity

### 5. Configuration Files
- `requirements.txt`: CPU-optimized PyTorch dependencies
- `runtime.txt`: Python version specification
- `render.yaml`: Deployment configuration
- `.gitignore`: Excludes large model files from Git

## 🎮 Usage Instructions

### Web Interface
1. **Visit**: [https://cod-769q.onrender.com](https://cod-769q.onrender.com) or run locally
2. **Upload**: Drag and drop an image or click "Browse Files"
3. **Analyze**: Click "Analyze Image" button
4. **View Results**: See detection, segmentation, and heatmap outputs
5. **Download**: Click on any result image to view full-screen and download

### API Endpoints
- `GET /`: Main web interface
- `POST /upload`: Image processing endpoint
- `GET /health`: Service health check
- `GET /style.css`: CSS stylesheet
- `GET /script.js`: JavaScript functionality

## 📊 Model Performance

- **Dataset**: COD10K (10,000+ camouflaged object images)
- **Architecture**: SINet V2 with Res2Net-50 backbone
- **Input Size**: 320×320 pixels
- **Inference Time**: ~2-3 seconds per image (CPU)
- **Confidence Threshold**: 0.01 (highly sensitive detection)

## 🌟 Key Innovations

1. **Multi-Scale Detection**: Handles objects of various sizes
2. **Attention Mechanisms**: Focuses on relevant image regions
3. **Neighbor Connection Decoder**: Improves boundary accuracy
4. **Group-Reversal Attention**: Enhances feature representation
5. **Real-time Processing**: Optimized for web deployment

## 🔄 Deployment Challenges & Solutions

### Challenge 1: Large Model Files
- **Problem**: Git can't handle large PyTorch models
- **Solution**: Automatic download from cloud storage during deployment

### Challenge 2: GPU to CPU Migration
- **Problem**: Local model trained on GPU, deployment on CPU
- **Solution**: Added `map_location='cpu'` and `weights_only=False` parameters

### Challenge 3: PyTorch Version Compatibility
- **Problem**: PyTorch 2.6 changed default loading behavior
- **Solution**: Explicit `weights_only=False` for backward compatibility

### Challenge 4: Cloud Storage Integration
- **Problem**: Google Drive blocks direct downloads
- **Solution**: Switched to Dropbox with `dl=1` parameter

## 🎯 Future Enhancements

- [ ] GPU acceleration for faster inference
- [ ] Batch processing for multiple images
- [ ] Video processing capabilities
- [ ] Mobile app development
- [ ] Advanced visualization options
- [ ] Custom model training interface

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is for educational and research purposes. Model weights and architecture are based on published research papers.

## 📞 Support

If you encounter any issues:
1. Check the [Issues](https://github.com/shashhii/COD/issues) page
2. Create a new issue with detailed description
3. Include error logs and system information

---

**Built with ❤️ using PyTorch, FastAPI, and deployed on Render**

⭐ **Star this repository if you found it helpful!**