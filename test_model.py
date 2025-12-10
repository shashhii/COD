import torch
import cv2
import numpy as np
import sys
from pathlib import Path

# Add Back End to path
sys.path.insert(0, str(Path(__file__).parent / "Back End"))
from sinetv2_model import SINetV2Model

async def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SINetV2Model(device)
    await model.load_model()
    print("Model loaded successfully")
    
    # Create a test image (white square on black background - easy to detect)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (200, 150), (440, 330), (255, 255, 255), -1)
    
    print(f"Test image shape: {test_image.shape}")
    
    # Test with very low threshold
    for threshold in [0.01, 0.05, 0.1, 0.3, 0.5]:
        detections = await model.predict(test_image, confidence_threshold=threshold)
        print(f"Threshold {threshold}: {len(detections)} detections")
        if detections:
            print(f"  First detection: {detections[0]}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_model())
