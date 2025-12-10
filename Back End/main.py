"""
SINet V2 COD Backend
"""

import os
import uuid
import asyncio
import torch
import cv2
import base64
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from sinetv2_model import SINetV2Model

# Initialize FastAPI app
app = FastAPI(title="COD API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    visualization: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    await model.load_model()
    print(f"SINet V2 loaded on {device}")

@app.get("/")
async def root():
    return {"message": "COD API is running", "status": "healthy"}

@app.post("/upload", response_model=ProcessingResponse)
async def upload_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        session_id = str(uuid.uuid4())
        upload_path = f"uploads/{session_id}_{file.filename}"
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process with SINet V2
        start_time = asyncio.get_event_loop().time()
        image = cv2.imread(upload_path)
        detections = await model.predict(image, confidence_threshold=0.5)
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Create results
        results = []
        for detection in detections:
            results.append(DetectionResult(
                object_name="Camouflaged Object",
                confidence=detection['confidence'],
                bbox=detection['bbox'],
                model_source="sinetv2"
            ))
        
        # Create visualization
        visualization = create_visualization(image, detections)
        
        response = ProcessingResponse(
            session_id=session_id,
            status="completed",
            results=results,
            processing_time=processing_time,
            visualization=visualization
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

def create_visualization(image, detections):
    try:
        result_image = image.copy()
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, f"{detection['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', result_image)
        return base64.b64encode(buffer).decode()
    except Exception:
        return ""

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)