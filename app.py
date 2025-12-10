import os
import uuid
import asyncio
import torch
import cv2
import base64
import numpy as np
import threading
import webbrowser
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Import SINet V2 model
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("sinetv2_model", Path(__file__).parent / "Back End" / "sinetv2_model.py")
if spec and spec.loader:
    sinetv2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sinetv2_module)
    SINetV2Model = sinetv2_module.SINetV2Model
else:
    raise ImportError("Could not load sinetv2_model")
from contextlib import asynccontextmanager

# Create directories
os.makedirs("uploads", exist_ok=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SINetV2Model(device)

class DetectionResult(BaseModel):
    object_name: str
    confidence: float
    bbox: List[float]
    model_source: str

class ProcessingResponse(BaseModel):
    session_id: str
    status: str
    results: List[DetectionResult]
    processing_time: float
    images: Dict[str, str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await model.load_model()
    print(f"SINet V2 loaded on {device}")
    yield
    # Shutdown
    pass

# Initialize FastAPI app
app = FastAPI(title="COD Application", version="1.0.0", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="Front End"), name="static")

@app.get("/")
async def root():
    return FileResponse("Front End/index.html", media_type="text/html")

@app.get("/style.css")
async def get_css():
    return FileResponse("Front End/style.css", media_type="text/css", headers={"Cache-Control": "no-cache"})

@app.get("/script.js")
async def get_js():
    return FileResponse("Front End/script.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"})

@app.get("/health")
async def health():
    return {"message": "COD API is running", "status": "healthy"}

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("Favicon.png", media_type="image/png")

@app.post("/upload", response_model=ProcessingResponse)
async def upload_image(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        session_id = str(uuid.uuid4())
        upload_path = os.path.join("uploads", f"{session_id}_{file.filename}")
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process with SINet V2
        start_time = asyncio.get_event_loop().time()
        image = cv2.imread(upload_path)
        print(f"Image loaded: {image.shape if image is not None else 'None'}")
        print(f"Upload path: {upload_path}")
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not load image")
            
        detections = await model.predict(image, confidence_threshold=0.01)
        print(f"Detections found: {len(detections)}")
        print(f"Detection details: {detections}")
        
        # Filter out very small objects (less than 1% of image area)
        h, w = image.shape[:2]
        min_area = (h * w) * 0.01
        detections = [d for d in detections if (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]) >= min_area]
        print(f"Detections after filtering small objects: {len(detections)}")
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Create results
        results = []
        for i, detection in enumerate(detections):
            results.append(DetectionResult(
                object_name=f"Camouflaged Object {i+1}",
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                model_source="sinetv2"
            ))
        
        # Create visualizations
        print("Creating visualizations...")
        detection_img = create_detection_visualization(image, detections)
        print(f"Detection image created: {len(detection_img) if detection_img else 0} chars")
        segmentation_img = create_segmentation_visualization(image, detections)
        print(f"Segmentation image created: {len(segmentation_img) if segmentation_img else 0} chars")
        heatmap_img = create_heatmap_visualization(image, detections)
        print(f"Heatmap image created: {len(heatmap_img) if heatmap_img else 0} chars")
        
        # If no visualizations created, use original image
        if not detection_img and image is not None:
            _, buffer = cv2.imencode('.jpg', image)
            detection_img = base64.b64encode(buffer).decode()
        if not segmentation_img and image is not None:
            _, buffer = cv2.imencode('.jpg', image)
            segmentation_img = base64.b64encode(buffer).decode()
        if not heatmap_img and image is not None:
            _, buffer = cv2.imencode('.jpg', image)
            heatmap_img = base64.b64encode(buffer).decode()
        
        response = ProcessingResponse(
            session_id=session_id,
            status="completed",
            results=results,
            processing_time=processing_time,
            images={
                "detection": detection_img,
                "segmentation": segmentation_img,
                "heatmap": heatmap_img
            }
        )
        
        print(f"Response created with {len(results)} results")
        print(f"Images in response: detection={len(detection_img) if detection_img else 0}, segmentation={len(segmentation_img) if segmentation_img else 0}, heatmap={len(heatmap_img) if heatmap_img else 0}")
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

def create_detection_visualization(image, detections):
    """Create bounding box visualization with different colors"""
    try:
        result_image = image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        if detections:
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                color = colors[i % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
                
                # Add label
                label = f"Object {i+1}: {detection['confidence']:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(result_image, (x1, y1-30), (x1+label_size[0], y1), color, -1)
                cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Add "No detections" text overlay
            h, w = result_image.shape[:2]
            text = "No camouflaged objects detected"
            font_scale = min(w, h) / 800
            thickness = max(1, int(font_scale * 2))
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            x = (w - text_size[0]) // 2
            y = h // 2
            cv2.rectangle(result_image, (x-10, y-text_size[1]-10), (x+text_size[0]+10, y+10), (0, 0, 0), -1)
            cv2.putText(result_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        _, buffer = cv2.imencode('.jpg', result_image)
        return base64.b64encode(buffer).decode()
    except Exception:
        return ""

def create_segmentation_visualization(image, detections):
    """Create mask overlay with different colors and boundaries"""
    try:
        result_image = image.copy()
        
        if detections:
            overlay = image.copy()
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            for i, detection in enumerate(detections):
                if 'mask' in detection and detection['mask']:
                    # Decode mask
                    import base64
                    from PIL import Image as PILImage
                    from io import BytesIO
                    
                    mask_data = base64.b64decode(detection['mask'])
                    mask_pil = PILImage.open(BytesIO(mask_data))
                    mask = np.array(mask_pil)
                    
                    # Resize mask to bbox size
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    mask_resized = cv2.resize(mask, (x2-x1, y2-y1))
                    
                    # Create colored mask
                    color = colors[i % len(colors)]
                    colored_mask = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
                    colored_mask[mask_resized > 127] = color
                    
                    # Apply to overlay
                    overlay[y1:y2, x1:x2][mask_resized > 127] = color
                    
                    # Create boundary
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        # Adjust contour coordinates
                        contour_adjusted = contour + [x1, y1]
                        cv2.drawContours(result_image, [contour_adjusted], -1, color, 2)
            
            # Blend with 50% opacity
            result_image = cv2.addWeighted(result_image, 0.5, overlay, 0.5, 0)
        else:
            # Add "No segmentation" text overlay
            h, w = result_image.shape[:2]
            text = "No objects to segment"
            font_scale = min(w, h) / 800
            thickness = max(1, int(font_scale * 2))
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            x = (w - text_size[0]) // 2
            y = h // 2
            cv2.rectangle(result_image, (x-10, y-text_size[1]-10), (x+text_size[0]+10, y+10), (0, 0, 0), -1)
            cv2.putText(result_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        
        _, buffer = cv2.imencode('.jpg', result_image)
        return base64.b64encode(buffer).decode()
    except Exception:
        return ""

def create_heatmap_visualization(image, detections):
    """Create thermal heatmap with gradient colors"""
    try:
        h, w = image.shape[:2]
        
        if detections:
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            for detection in detections:
                if 'mask' in detection and detection['mask']:
                    from PIL import Image as PILImage
                    from io import BytesIO
                    
                    mask_data = base64.b64decode(detection['mask'])
                    mask_pil = PILImage.open(BytesIO(mask_data))
                    mask = np.array(mask_pil)
                    
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    mask_resized = cv2.resize(mask, (x2-x1, y2-y1))
                    
                    mask_binary = (mask_resized > 127).astype(np.uint8)
                    dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
                    
                    if dist_transform.max() > 0:
                        gradient = dist_transform / dist_transform.max()
                        # Scale to full range 0.3-1.0 for richer colors
                        gradient = 0.3 + gradient * 0.7
                    else:
                        gradient = mask_binary.astype(np.float32)
                    
                    heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], gradient)
            
            # Apply Gaussian blur for smoother gradients
            heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
            
            # Normalize to full 0-255 range
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            heatmap_normalized = (heatmap * 255).astype(np.uint8)
            
            # Apply JET colormap
            result_image = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            
            # Blend with original
            result_image = cv2.addWeighted(image, 0.3, result_image, 0.7, 0)
        else:
            result_image = image.copy()
            text = "No probability regions detected"
            font_scale = min(w, h) / 800
            thickness = max(1, int(font_scale * 2))
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            x = (w - text_size[0]) // 2
            y = h // 2
            cv2.rectangle(result_image, (x-10, y-text_size[1]-10), (x+text_size[0]+10, y+10), (0, 0, 0), -1)
            cv2.putText(result_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), thickness)
        
        _, buffer = cv2.imencode('.jpg', result_image)
        return base64.b64encode(buffer).decode()
    except Exception:
        return ""

def start_server():
    """Start the FastAPI server"""
    uvicorn.run(app, host="localhost", port=8000, log_level="info")

def open_browser():
    """Open browser after server starts"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")